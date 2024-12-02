using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChurnPrediction
{
    public class CustomerData
    {
        public float Usage { get; set; }
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
                .Append(context.Transforms.Concatenate("Features", nameof(CustomerData.Usage)))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(splitData.TrainSet);

            // Evaluate the model
            var predictions = model.Transform(splitData.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.AccuracyMacro}");

            // Use the model for predictions
            var predictionEngine = context.Model.CreatePredictionEngine<CustomerData, CustomerPrediction>(model);

            var newCustomer = new CustomerData { Usage = 3.5f };
            var prediction = predictionEngine.Predict(newCustomer);

            Console.WriteLine($"Predicted Churn: {prediction.Churn}");
        }
    }
}
