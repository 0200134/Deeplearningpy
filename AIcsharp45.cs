using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Neurons;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.Controls;
using Accord.Math.Random;
using Accord.Imaging.Filters;

namespace CNNImageClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load and preprocess the dataset
            string datasetPath = "path_to_your_dataset";
            var data = LoadDataset(datasetPath);
            var (trainImages, trainLabels, testImages, testLabels) = SplitDataset(data, 0.8);

            // Initialize CNN
            var cnn = new ConvolutionalNeuralNetwork(
                new ConvolutionalLayer(28, 28, 1, 16, 3, 3),
                new SubsamplingLayer(2, 2),
                new ConvolutionalLayer(13, 13, 16, 32, 3, 3),
                new SubsamplingLayer(2, 2),
                new ConvolutionalLayer(5, 5, 32, 64, 3, 3),
                new SubsamplingLayer(2, 2),
                new FullyConnectedLayer(128),
                new SoftmaxLayer(10));

            cnn.Randomize();

            // Train the CNN
            var teacher = new ParallelResilientBackpropagationLearning(cnn)
            {
                BatchSize = 100,
                MaxIterations = 1000,
                Tolerance = 1e-5
            };

            double error = double.PositiveInfinity;
            while (error > 0.01)
            {
                error = teacher.RunEpoch(trainImages, trainLabels);
                Console.WriteLine($"Training error: {error}");
            }

            // Evaluate the CNN
            int correct = 0;
            for (int i = 0; i < testImages.Length; i++)
            {
                var output = cnn.Compute(testImages[i]);
                if (output.ArgMax() == testLabels[i].ArgMax())
                {
                    correct++;
                }
            }
            double accuracy = (double)correct / testImages.Length;
            Console.WriteLine($"Test accuracy: {accuracy}");

            // Save the model
            cnn.Save("cnn_model.bin");
        }

        static (double[][], double[][], double[][], double[][]) SplitDataset((double[][] images, double[][] labels) data, double trainRatio)
        {
            int trainSize = (int)(data.images.Length * trainRatio);
            int testSize = data.images.Length - trainSize;

            var trainImages = new double[trainSize][];
            var trainLabels = new double[trainSize][];
            var testImages = new double[testSize][];
            var testLabels = new double[testSize][];

            Array.Copy(data.images, trainImages, trainSize);
            Array.Copy(data.labels, trainLabels, trainSize);
            Array.Copy(data.images, trainSize, testImages, 0, testSize);
            Array.Copy(data.labels, trainSize, testLabels, 0, testSize);

            return (trainImages, trainLabels, testImages, testLabels);
        }

        static (double[][] images, double[][] labels) LoadDataset(string datasetPath)
        {
            var images = new List<double[]>();
            var labels = new List<double[]>();

            foreach (var file in Directory.GetFiles(datasetPath, "*.jpg"))
            {
                var image = new Bitmap(file);
                var resizedImage = new ResizeBilinear(28, 28).Apply(image);
                images.Add(ImageToArray(resizedImage));
                labels.Add(GetLabelFromFileName(file));
            }

            return (images.ToArray(), labels.ToArray());
        }

        static double[] ImageToArray(Bitmap image)
        {
            double[] array = new double[image.Width * image.Height];
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    array[y * image.Width + x] = image.GetPixel(x, y).R / 255.0;
                }
            }
            return array;
        }

        static double[] GetLabelFromFileName(string fileName)
        {
            int label = int.Parse(Path.GetFileNameWithoutExtension(fileName).Split('_')[1]);
            double[] oneHotLabel = new double[10];
            oneHotLabel[label] = 1.0;
            return oneHotLabel;
        }
    }
}
