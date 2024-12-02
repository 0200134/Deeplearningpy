using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetImageClassification
{
    class Program
    {
        // Define data structure for images
        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        public class ImagePrediction : ImageData
        {
            public float[] Score;
            public string PredictedLabel;
        }

        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<ImageData>("images.tsv", hasHeader: false);

            var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.LoadImages("ImagePath", "ImagePath"))
                .Append(context.Transforms.ResizeImages("ImagePath", 224, 224))
                .Append(context.Model.LoadTensorFlowModel("model/tensorflow_inception_graph.pb")
                    .ScoreTensorName("softmax"))
                .Append(context.Transforms.CopyColumns("PredictedLabel", "Label"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            var prediction = predictionEngine.Predict(new ImageData
            {
                ImagePath = "path/to/your/image.jpg"
            });

            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
        }
    }
}
