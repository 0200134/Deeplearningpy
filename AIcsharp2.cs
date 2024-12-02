using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;

namespace ImageClassification
{
    public class ImageData
    {
        [ImageType(224, 224)]
        public Bitmap Image { get; set; }

        [LoadColumn(0)]
        public string Label { get; set; }
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score { get; set; }
        public string PredictedLabel { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load and transform data
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageData>());
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadImages("ImagePath", nameof(ImageData.Image)))
                .Append(mlContext.Transforms.ResizeImages("ImageResized", 224, 224, nameof(ImageData.Image)))
                .Append(mlContext.Transforms.ExtractPixels("Input", "ImageResized"))
                .Append(mlContext.Model.LoadTensorFlowModel("model.onnx")
                    .ScoreTensorName("output")
                    .AddInput("Input"));

            // Train the model
            var model = pipeline.Fit(data);

            // Save the model
            mlContext.Model.Save(model, data.Schema, "model.zip");

            // Load and use the model for predictions
            var loadedModel = mlContext.Model.Load("model.zip", out var schema);
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

            // Predict
            var testImage = new ImageData { Image = new Bitmap("path_to_test_image.jpg") };
            var prediction = predictor.Predict(testImage);
            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
        }
    }
}
