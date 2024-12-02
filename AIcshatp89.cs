using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace MultiTaskLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            // Sample text data
            var texts = new string[]
            {
                "I love programming!",
                "TensorFlow is great.",
                "C# is a powerful language.",
                "I enjoy learning about AI.",
                "This is an advanced topic in AI."
            };
            var textLabels = new int[] { 1, 1, 1, 1, 1 }; // Example labels for simplicity

            // Sample image data (28x28 grayscale images)
            var images = GenerateSampleImageData(1000, 28, 28);
            var imageLabels = new int[] { 1, 0, 1, 0, 1 }; // Example labels for simplicity

            // Preprocess the data
            var (textInputIds, textAttentionMasks) = TokenizeAndPreprocess(texts);
            var (imageTrainX, imageTrainY) = SplitDataset(images, imageLabels, 0.8);

            // Build and compile the multi-task learning model
            var model = BuildMultiTaskModel();
            model.compile(optimizer: new Adam(0.0001), loss: new Dictionary<string, string>
            {
                { "text_output", "sparse_categorical_crossentropy" },
                { "image_output", "sparse_categorical_crossentropy" }
            }, metrics: new Dictionary<string, string[]>
            {
                { "text_output", new[] { "accuracy" } },
                { "image_output", new[] { "accuracy" } }
            });

            // Train the model
            model.fit(new Dictionary<string, NDArray>
            {
                { "text_input_ids", textInputIds },
                { "image_inputs", imageTrainX }
            }, new Dictionary<string, NDArray>
            {
                { "text_output", np.array(textLabels) },
                { "image_output", np.array(imageLabels) }
            }, batch_size: 32, epochs: 10, validation_split: 0.2);

            // Evaluate the model
            var results = model.evaluate(new Dictionary<string, NDArray>
            {
                { "text_input_ids", textInputIds },
                { "image_inputs", imageTrainX }
            }, new Dictionary<string, NDArray>
            {
                { "text_output", np.array(textLabels) },
                { "image_output", np.array(imageLabels) }
            });
            Console.WriteLine($"Test loss: {results[0]}, Test accuracy: {results[1]}");
        }

        static (NDArray, NDArray) TokenizeAndPreprocess(string[] texts)
        {
            var tokenizer = new BertTokenizer("bert-base-uncased");

            List<long[]> inputIdsList = new List<long[]>();
            List<long[]> attentionMasksList = new List<long[]>();

            foreach (var text in texts)
            {
                var tokens = tokenizer.Encode(text);
                inputIdsList.Add(tokens.InputIds);
                attentionMasksList.Add(tokens.AttentionMask);
            }

            var inputIds = np.array(inputIdsList);
            var attentionMasks = np.array(attentionMasksList);
            return (inputIds, attentionMasks);
        }

        static (NDArray, NDArray) GenerateSampleImageData(int numSamples, int width, int height)
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

        static (NDArray, NDArray) SplitDataset(NDArray data, int[] labels, double trainRatio)
        {
            int trainSize = (int)(data.shape[0] * trainRatio);
            int testSize = data.shape[0] - trainSize;

            var trainData = data[$"{0}:{trainSize}", ":"];
            var trainLabels = labels[0..trainSize];
            return (trainData, np.array(trainLabels));
        }

        static Model BuildMultiTaskModel()
        {
            var textInputIds = keras.Input(shape: (128,), dtype: tf.int32, name: "text_input_ids");
            var imageInputs = keras.Input(shape: (28, 28, 1), dtype: tf.float32, name: "image_inputs");

            // Text encoder using BERT
            var bertLayer = new TFBertModel("bert-base-uncased").call()[0];
            var textOutputs = bertLayer(textInputIds)[0];
            var textCLS = textOutputs[:, 0, :];
            var textDense = new Dense(64, activation: "relu").Apply(textCLS);
            var textOutput = new Dense(2, activation: "softmax", name: "text_output").Apply(textDense);

            // Image encoder using CNN
            var x = new Conv2D(32, (3, 3), activation: "relu").Apply(imageInputs);
            x = new MaxPooling2D((2, 2)).Apply(x);
            x = new Conv2D(64, (3, 3), activation: "relu").Apply(x);
            x = new MaxPooling2D((2, 2)).Apply(x);
            x = new Flatten().Apply(x);
            var imageDense = new Dense(64, activation: "relu").Apply(x);
            var imageOutput = new Dense(2, activation: "softmax", name: "image_output").Apply(imageDense);

            // Combined model
            var model = keras.Model(new Tensors(textInputIds, imageInputs), new Tensors(textOutput, imageOutput));
            model.summary();
            return model;
        }
    }
}
