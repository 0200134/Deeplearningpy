using System;
using Tensorflow;
using static Tensorflow.Binding;
using NumSharp;

namespace ImageClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load MNIST dataset
            var mnist = MnistModelLoader.LoadMnist();

            // Define placeholders for input images and labels
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 28, 28, 1), name: "x");
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10), name: "y");

            // Define the neural network
            var conv1 = tf.layers.conv2d(x, filters: 32, kernel_size: 5, activation: tf.nn.relu);
            var pool1 = tf.layers.max_pooling2d(conv1, pool_size: 2, strides: 2);
            var conv2 = tf.layers.conv2d(pool1, filters: 64, kernel_size: 5, activation: tf.nn.relu);
            var pool2 = tf.layers.max_pooling2d(conv2, pool_size: 2, strides: 2);
            var flatten = tf.layers.flatten(pool2);
            var fc1 = tf.layers.dense(flatten, units: 1024, activation: tf.nn.relu);
            var logits = tf.layers.dense(fc1, units: 10);

            // Define loss and optimizer
            var loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels: y, logits: logits));
            var optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss);

            // Accuracy metric
            var correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1));
            var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

            // Initialize session
            var session = tf.Session();
            session.run(tf.global_variables_initializer());

            // Train the model
            for (int i = 0; i < 1000; i++)
            {
                var batch = mnist.train.next_batch(50);
                session.run(optimizer, (x, batch.Data), (y, batch.Labels));

                if (i % 100 == 0)
                {
                    var train_accuracy = session.run(accuracy, (x, batch.Data), (y, batch.Labels));
                    Console.WriteLine($"Step {i}, Training Accuracy: {train_accuracy}");
                }
            }

            // Evaluate the model
            var test_accuracy = session.run(accuracy, (x, mnist.test.Data), (y, mnist.test.Labels));
            Console.WriteLine($"Test Accuracy: {test_accuracy}");
        }
    }
}
