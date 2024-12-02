using System;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace VariationalAutoencoder
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load and preprocess dataset
            var data = GenerateSampleData(1000, 28, 28);
            var (trainX, testX) = SplitDataset(data, 0.8);

            // Build VAE model
            var (vae, encoder, decoder) = BuildVAEModel(inputShape: (28, 28, 1), latentDim: 2);

            // Compile VAE model
            vae.compile(optimizer: new Adam(0.001), loss: ComputeVAELoss);

            // Train VAE model
            vae.fit(trainX, trainX, batch_size: 32, epochs: 50, validation_split: 0.2);

            // Evaluate VAE model
            var testLoss = vae.evaluate(testX, testX);
            Console.WriteLine($"Test loss: {testLoss}");

            // Generate new data points
            var randomSample = np.random.normal(0, 1, new Shape(1, 2));
            var generatedImage = decoder.predict(randomSample);
            Console.WriteLine($"Generated image shape: {generatedImage.shape}");
        }

        static NDArray GenerateSampleData(int numSamples, int width, int height)
        {
            var rand = new Random();
            var data = new NDArray(new Shape(numSamples, width, height, 1));
            for (int i = 0; i < numSamples; i++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        data[i, x, y, 0] = rand.NextDouble();
                    }
                }
            }
            return data;
        }

        static (NDArray, NDArray) SplitDataset(NDArray data, double trainRatio)
        {
            int trainSize = (int)(data.shape[0] * trainRatio);
            int testSize = data.shape[0] - trainSize;

            var trainData = data[$"{0}:{trainSize}", ":"];
            var testData = data[$"{trainSize}:{data.shape[0]}", ":"];
            return (trainData, testData);
        }

        static (Model, Model, Model) BuildVAEModel((int, int, int) inputShape, int latentDim)
        {
            // Encoder
            var encoderInputs = keras.Input(shape: inputShape);
            var x = new Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(encoderInputs);
            x = new MaxPooling2D((2, 2), padding: "same").Apply(x);
            x = new Conv2D(64, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = new MaxPooling2D((2, 2), padding: "same").Apply(x);
            x = new Flatten().Apply(x);
            var zMean = new Dense(latentDim).Apply(x);
            var zLogVar = new Dense(latentDim).Apply(x);

            var encoder = keras.Model(encoderInputs, new Tensors(zMean, zLogVar), name: "encoder");

            // Sampling function
            Func<Tensor, Tensor, Tensor> Sampling = (zMean, zLogVar) =>
            {
                var epsilon = keras.backend.random_normal(zMean.shape);
                return zMean + keras.backend.exp(zLogVar * 0.5) * epsilon;
            };
            var z = new Lambda(Sampling).Apply(new Tensors(zMean, zLogVar));

            // Decoder
            var latentInputs = keras.Input(shape: latentDim);
            x = new Dense(7 * 7 * 64, activation: "relu").Apply(latentInputs);
            x = new Reshape((7, 7, 64)).Apply(x);
            x = new Conv2DTranspose(64, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = new UpSampling2D((2, 2)).Apply(x);
            x = new Conv2DTranspose(32, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = new UpSampling2D((2, 2)).Apply(x);
            var decoderOutputs = new Conv2DTranspose(1, (3, 3), activation: "sigmoid", padding: "same").Apply(x);

            var decoder = keras.Model(latentInputs, decoderOutputs, name: "decoder");

            // VAE
            var vaeOutputs = decoder.Apply(z);
            var vae = keras.Model(encoderInputs, vaeOutputs, name: "vae");

            return (vae, encoder, decoder);
        }

        static Tensor ComputeVAELoss(Tensor yTrue, Tensor yPred)
        {
            var reconstructionLoss = keras.losses.binary_crossentropy(yTrue, yPred);
            reconstructionLoss = keras.backend.sum(reconstructionLoss, axis: new int[] { 1, 2, 3 });

            var klLoss = -0.5 * keras.backend.sum(1 + zLogVar - keras.backend.square(zMean) - keras.backend.exp(zLogVar), axis: -1);
            return keras.backend.mean(reconstructionLoss + klLoss);
        }
    }
}
