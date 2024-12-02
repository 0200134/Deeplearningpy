using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChurnPrediction
{
    public class CustomerData
    {
        public float Usage { get; set; }
        public float Age { get; set; }
        public float Income { get; set; }
        public bool Churn { get; set; }
    }

    public class CustomerPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Churn { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data
            var data = context.Data.LoadFromTextFile<CustomerData>("data.csv", separatorChar: ',', hasHeader: true);

            // Split data into training and testing sets
            var splitData = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define the data processing and model pipeline
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label", nameof(CustomerData.Churn))
                .Append(context.Transforms.Concatenate("Features", nameof(CustomerData.Usage), nameof(CustomerData.Age), nameof(CustomerData.Income)))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(splitData.TrainSet);

            // Save the model
            context.Model.Save(model, splitData.TrainSet.Schema, "model.zip");
            Console.WriteLine("Model saved.");

            // Load the model
            ITransformer loadedModel;
            DataViewSchema modelSchema;
            using (var stream = new FileStream("model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = context.Model.Load(stream, out modelSchema);
            }

            // Evaluate the model
            var predictions = model.Transform(splitData.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {metrics.AccuracyMacro}");

            // Use the model for predictions
            var predictionEngine = context.Model.CreatePredictionEngine<CustomerData, CustomerPrediction>(model);
            var newCustomer = new CustomerData { Usage = 3.5f, Age = 30, Income = 50000 };
            var prediction = predictionEngine.Predict(newCustomer);
            Console.WriteLine($"Predicted Churn: {prediction.Churn}");

            // Use the loaded model for predictions
            var loadedPredictionEngine = context.Model.CreatePredictionEngine<CustomerData, CustomerPrediction>(loadedModel);
            var loadedPrediction = loadedPredictionEngine.Predict(newCustomer);
            Console.WriteLine($"Loaded Model Predicted Churn: {loadedPrediction.Churn}");
        }
    }
}
