using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace TextClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load data
            var data = new List<TextData>
            {
                new TextData { Text = "This is a positive text", Label = true },
                new TextData { Text = "This is a negative text", Label = false },
                // Add more data here
            };

            var trainingData = mlContext.Data.LoadFromEnumerable(data);

            // Define the text classification pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(nameof(TextData.Label)))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.Concatenate("Features", "Features"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree());

            // Train the model
            var model = pipeline.Fit(trainingData);

            // Use the model for prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TextPrediction>(model);

            var sampleData = new TextData { Text = "This is an amazing day!" };
            var prediction = predictionEngine.Predict(sampleData);
            Console.WriteLine($"Predicted label: {(prediction.PredictedLabel ? "Positive" : "Negative")}, Score: {prediction.Score}");
        }

        public class TextData
        {
            public string Text { get; set; }
            public bool Label { get; set; }
        }

        public class TextPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel { get; set; }

            public float Score { get; set; }
        }
    }
}
