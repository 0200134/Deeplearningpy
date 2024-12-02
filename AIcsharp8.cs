using System;

namespace NeuralNetworkExample
{
    class NeuralNetwork
    {
        private int[] layers;
        private float[][] neurons;
        private float[][][] weights;
        private Random rand;

        public NeuralNetwork(int[] layers)
        {
            this.layers = layers;
            rand = new Random();

            InitNeurons();
            InitWeights();
        }

        private void InitNeurons()
        {
            var neuronsList = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                neuronsList[i] = new float[layers[i]];
            }
            neurons = neuronsList;
        }

        private void InitWeights()
        {
            var weightsList = new float[layers.Length - 1][][];
            for (int i = 0; i < layers.Length - 1; i++)
            {
                var layerWeights = new float[neurons[i + 1].Length][];
                for (int j = 0; j < neurons[i + 1].Length; j++)
                {
                    layerWeights[j] = new float[neurons[i].Length];
                    for (int k = 0; k < neurons[i].Length; k++)
                    {
                        layerWeights[j][k] = (float)rand.NextDouble() - 0.5f;
                    }
                }
                weightsList[i] = layerWeights;
            }
            weights = weightsList;
        }

        public float[] FeedForward(float[] inputs)
        {
            Array.Copy(inputs, neurons[0], inputs.Length);

            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float value = 0f;
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        value += weights[i - 1][j][k] * neurons[i - 1][k];
                    }
                    neurons[i][j] = (float)Math.Tanh(value);
                }
            }
            return neurons[neurons.Length - 1];
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            int[] layers = { 2, 3, 1 }; // Example with 2 input neurons, 3 hidden neurons, and 1 output neuron
            NeuralNetwork nn = new NeuralNetwork(layers);

            float[] inputs = { 0.5f, -0.5f };
            float[] output = nn.FeedForward(inputs);

            Console.WriteLine($"Output: {output[0]}");
        }
    }
}
