using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace BERTTextClassification
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
            var labels = new int[] { 1, 1, 1, 1, 1 }; // Example labels for simplicity

            // Preprocess the data
            var (inputIds, attentionMasks) = TokenizeAndPreprocess(texts);

            // Build and train the BERT model
            var model = BuildBERTModel();
            model.compile(optimizer: new Adam(2e-5), loss: "binary_crossentropy", metrics: new[] { "accuracy" });

            // Training
            model.fit(inputIds, labels, batch_size: 2, epochs: 3, validation_split: 0.2);

            // Evaluate the model
            var testLoss = model.evaluate(inputIds, labels);
            Console.WriteLine($"Test loss: {testLoss[0]}, Test accuracy: {testLoss[1]}");
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
    }
}static Model BuildBERTModel()
{
    var bertLayer = new TFBertModel("bert-base-uncased").call()[0];
    var inputIds = tf.keras.Input(shape: (128,), dtype: tf.int32, name: "input_ids");
    var attentionMask = tf.keras.Input(shape: (128,), dtype: tf.int32, name: "attention_mask");

    var outputs = bertLayer(inputIds, attention_mask: attentionMask)[0];
    var clsToken = outputs[:, 0, :]; // Take the CLS token's output
    var output = new Dense(1, activation: "sigmoid").Apply(clsToken);

    var model = tf.keras.Model(new Tensors(inputIds, attentionMask), output);
    model.summary();
    return model;
}
