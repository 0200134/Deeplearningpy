using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace TransformerExample
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load the data
            var dataPath = "path/to/your/data.csv";
            var dataView = mlContext.Data.LoadFromTextFile<InputData>(dataPath, separatorChar: ',', hasHeader: true);

            // Define the pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Text", "Features")
                .Append(mlContext.Transforms.Text.NormalizeText("Text", "NormalizedText"))
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("NormalizedText", "Tokens"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Tokens", "Embeddings", WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding))
                .Append(mlContext.Transforms.Concatenate("Features", "Embeddings"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.Transforms.Text.FeaturizeText("Text", "Features"))
                .Append(mlContext.Regression.Trainers.Sdca());

            // Train the model
            var model = pipeline.Fit(dataView);

            // Make predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);
            var prediction = predictionEngine.Predict(new InputData { Text = "Your text here" });

            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
        }

        public class InputData
        {
            [LoadColumn(0)]
            public string Text { get; set; }

            [LoadColumn(1)]
            public string Label { get; set; }
        }

        public class OutputData
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; }
        }
    }
}
