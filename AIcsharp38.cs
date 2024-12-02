using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace LSTMTimeSeriesForecasting
{
    class Program
    {
        static void Main(string[] args)
        {
            // Generate synthetic time series data
            var data = GenerateTimeSeriesData(1000);

            // Preprocess data
            var (trainX, trainY, testX, testY) = PreprocessData(data, 60);

            // Initialize and build LSTM model
            var model = BuildModel(inputShape: (trainX.shape[1], trainX.shape[2]));

            // Compile model
            model.compile(optimizer: new Adam(0.001), loss: "mse");

            // Train model
            model.fit(trainX, trainY, batch_size: 32, epochs: 50, validation_split: 0.2);

            // Evaluate model
            var testLoss = model.evaluate(testX, testY);
            Console.WriteLine($"Test loss: {testLoss}");

            // Forecast future values
            var forecast = model.predict(testX);
            Console.WriteLine($"Forecasted values: {string.Join(", ", forecast)}");
        }

        static NDArray GenerateTimeSeriesData(int numSteps)
        {
            var data = new NDArray(new Shape(numSteps, 1));
            var rand = new Random();
            for (int i = 0; i < numSteps; i++)
            {
                data[i] = Math.Sin(i * 0.01) + rand.NextDouble() * 0.1;
            }
            return data;
        }

        static (NDArray, NDArray, NDArray, NDArray) PreprocessData(NDArray data, int timeSteps)
        {
            var X = new List<NDArray>();
            var Y = new List<NDArray>();

            for (int i = 0; i < data.shape[0] - timeSteps; i++)
            {
                var x = data[$"{i}:{i + timeSteps}"];
                var y = data[i + timeSteps];
                X.Add(x);
                Y.Add(y);
            }

            var XArray = np.array(X);
            var YArray = np.array(Y);
            var splitIndex = (int)(XArray.shape[0] * 0.8);

            return (XArray[":", ":"].Slice(new Slice(0, splitIndex)),
                    YArray.Slice(new Slice(0, splitIndex)),
                    XArray[":", ":"].Slice(new Slice(splitIndex, XArray.shape[0])),
                    YArray.Slice(new Slice(splitIndex, YArray.shape[0])));
        }

        static Model BuildModel((int, int) inputShape)
        {
            var inputs = keras.Input(shape: inputShape);
            var x = new LSTM(50, return_sequences: true).Apply(inputs);
            x = new LSTM(50, return_sequences: false).Apply(x);
            x = new Dense(1).Apply(x);

            var model = keras.Model(inputs, x);
            model.summary();
            return model;
        }
    }
}
