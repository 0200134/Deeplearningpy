using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace RecommendationSystem
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load data
            var data = mlContext.Data.LoadFromTextFile<MovieRating>("ratings.csv", hasHeader: true, separatorChar: ',');

            // Split data into training and testing sets
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Define the recommendation pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("movieId"))
                .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options =>
                {
                    options.MatrixColumnIndexColumnName = "userId";
                    options.MatrixRowIndexColumnName = "movieId";
                    options.LabelColumnName = "Label";
                }));

            // Train the model
            var model = pipeline.Fit(trainingData);

            // Evaluate the model
            var prediction = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            // Use the model for prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            var testInput = new MovieRating { userId = "6", movieId = "10" };
            var ratingPrediction = predictionEngine.Predict(testInput);
            Console.WriteLine($"Predicted rating for user 6 and movie 10: {ratingPrediction.Score}");
        }
    }

    // Data model for loading movie ratings
    public class MovieRating
    {
        [LoadColumn(0)]
        public string userId;
        [LoadColumn(1)]
        public string movieId;
        [LoadColumn(2)]
        public float Label;
    }

    // Class for storing the prediction
    public class MovieRatingPrediction
    {
        public float Score { get; set; }
    }
}
