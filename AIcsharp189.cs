using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace AdvancedRecommendationSystem
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load the data
            string dataPath = "path/to/your/dataset.csv";
            var dataView = mlContext.Data.LoadFromTextFile<RatingData>(dataPath, hasHeader: true, separatorChar: ',');

            // Split the data into training and test sets
            var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = splitData.TrainSet;
            var testData = splitData.TestSet;

            // Define the data preprocessing pipeline
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("UserId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("MovieId"));

            // Define the matrix factorization model
            var matrixFactorizationTrainer = mlContext.Recommendation().Trainers.MatrixFactorization(
                labelColumnName: "Label",
                matrixColumnIndexColumnName: "UserIdEncoded",
                matrixRowIndexColumnName: "MovieIdEncoded");

            // Train the model
            var trainingPipeline = dataProcessPipeline.Append(matrixFactorizationTrainer);
            var model = trainingPipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            // Create the prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<RatingData, RatingPrediction>(model);

            // Make a recommendation
            var userId = "user_id_here";
            var movieId = "movie_id_here";
            var prediction = predictionEngine.Predict(new RatingData { UserId = userId, MovieId = movieId });

            Console.WriteLine($"Predicted rating for user {userId} and movie {movieId}: {prediction.Score}");
        }

        public class RatingData
        {
            public string UserId { get; set; }
            public string MovieId { get; set; }
            public float Label { get; set; }
        }

        public class RatingPrediction
        {
            public float Score { get; set; }
        }
    }
}
