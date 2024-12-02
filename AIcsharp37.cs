using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Losses;
using NumSharp;

namespace DDQN
{
    class Program
    {
        static void Main(string[] args)
        {
            var env = new CartPoleEnvironment();
            int stateDim = env.StateDim;
            int actionDim = env.ActionDim;
            int batchSize = 64;
            int memorySize = 10000;
            int targetUpdateFrequency = 1000;
            double gamma = 0.99;
            double lr = 0.001;
            int episodes = 1000;

            var agent = new DDQNAgent(stateDim, actionDim, memorySize, batchSize, gamma, lr, targetUpdateFrequency);
            var replayBuffer = new ReplayBuffer(memorySize, batchSize);

            for (int episode = 0; episode < episodes; episode++)
            {
                var state = env.Reset();
                double episodeReward = 0;
                bool done = false;

                while (!done)
                {
                    var action = agent.GetAction(state);
                    var (nextState, reward, done) = env.Step(action);

                    replayBuffer.Add(state, action, reward, nextState, done);
                    state = nextState;
                    episodeReward += reward;

                    if (replayBuffer.Size >= batchSize)
                    {
                        var (states, actions, rewards, nextStates, dones) = replayBuffer.Sample();
                        agent.Train(states, actions, rewards, nextStates, dones);
                    }

                    if (done)
                    {
                        Console.WriteLine($"Episode {episode + 1}, Reward: {episodeReward}");
                        break;
                    }
                }

                if ((episode + 1) % targetUpdateFrequency == 0)
                {
                    agent.UpdateTargetNetwork();
                }
            }
        }
    }

    public class DDQNAgent
    {
        private readonly int stateDim;
        private readonly int actionDim;
        private readonly int memorySize;
        private readonly int batchSize;
        private readonly double gamma;
        private readonly double lr;
        private readonly int targetUpdateFrequency;

        private readonly Model qNetwork;
        private readonly Model targetNetwork;
        private readonly Optimizer optimizer;

        public DDQNAgent(int stateDim, int actionDim, int memorySize, int batchSize, double gamma, double lr, int targetUpdateFrequency)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.memorySize = memorySize;
            this.batchSize = batchSize;
            this.gamma = gamma;
            this.lr = lr;
            this.targetUpdateFrequency = targetUpdateFrequency;

            this.qNetwork = BuildModel();
            this.targetNetwork = BuildModel();
            this.targetNetwork.SetWeights(this.qNetwork.GetWeights());
            this.optimizer = tf.keras.optimizers.Adam(lr);
        }

        private Model BuildModel()
        {
            var inputs = tf.keras.Input(shape: stateDim);
            var dense1 = new Dense(64, activation: "relu").Apply(inputs);
            var dense2 = new Dense(64, activation: "relu").Apply(dense1);
            var outputs = new Dense(actionDim).Apply(dense2);
            var model = tf.keras.Model(inputs, outputs);
            model.Compile(optimizer: optimizer, loss: tf.keras.losses.MeanSquaredError());
            return model;
        }

        public int GetAction(float[] state, double epsilon = 0.1)
        {
            if (np.random.rand() < epsilon)
            {
                return new Random().Next(actionDim);
            }
            var qValues = qNetwork.Predict(state);
            return np.argmax(qValues);
        }

        public void Train(NDArray states, NDArray actions, NDArray rewards, NDArray nextStates, NDArray dones)
        {
            var nextQValues = targetNetwork.Predict(nextStates);
            var nextQValue = np.max(nextQValues, axis: 1);
            var targets = rewards + gamma * nextQValue * (1 - dones);

            var masks = np.one_hot(actions, actionDim);
            var withMask = masks * qNetwork.Predict(states);
            var maskedTargets = np.array([withMask[i] + (targets[i] - withMask[i]) for i in range(batchSize)]);

            qNetwork.TrainOnBatch(states, maskedTargets);
        }

        public void UpdateTargetNetwork()
        {
            targetNetwork.SetWeights(qNetwork.GetWeights());
        }
    }

    public class ReplayBuffer
    {
        private readonly int memorySize;
        private readonly int batchSize;
        private int position;
        private readonly List<float[]> states;
        private readonly List<int> actions;
        private readonly List<float> rewards;
        private readonly List<float[]> nextStates;
        private readonly List<bool> dones;

        public ReplayBuffer(int memorySize, int batchSize)
        {
            this.memorySize = memorySize;
            this.batchSize = batchSize;
            this.position = 0;
            this.states = new List<float[]>(memorySize);
            this.actions = new List<int>(memorySize);
            this.rewards = new List<float>(memorySize);
            this.nextStates = new List<float[]>(memorySize);
            this.dones = new List<bool>(memorySize);
        }

        public void Add(float[] state, int action, float reward, float[] nextState, bool done)
        {
            if (states.Count < memorySize)
            {
                states.Add(state);
                actions.Add(action);
                rewards.Add(reward);
                nextStates.Add(nextState);
                dones.Add(done);
            }
            else
            {
                states[position] = state;
                actions[position] = action;
                rewards[position] = reward;
                nextStates[position] = nextState;
                dones[position] = done;
            }
            position = (position + 1) % memorySize;
        }

        public (NDArray states, NDArray actions, NDArray rewards, NDArray nextStates, NDArray dones) Sample()
        {
            var indices = np.random.randint(0, states.Count, batchSize);
            var sampleStates = new List<float[]>();
            var sampleActions = new List<int>();
            var sampleRewards = new List<float>();
            var sampleNextStates = new List<float[]>();
            var sampleDones = new List<bool>();

            foreach (var idx in indices)
            {
                sampleStates.Add(states[idx]);
                sampleActions.Add(actions[idx]);
                sampleRewards.Add(rewards[idx]);
                sampleNextStates.Add(nextStates[idx]);
                sampleDones.Add(dones[idx]);
            }

            return (np.array(sampleStates), np.array(sampleActions), np.array(sampleRewards), np.array(sampleNextStates), np.array(sampleDones));
        }

        public int Size => states.Count;
    }

    public class CartPoleEnvironment
    {
        private Random random;
        private float[] state;

        public int StateDim => 4;
        public int ActionDim => 2;

        public CartPoleEnvironment()
        {
            random = new Random();
            state = new float[StateDim];
        }

        public float[] Reset()
        {
            for (int i = 0; i < StateDim; i++)
            {
                state[i] = (float)random.NextDouble();
            }
            return state;
        }

        public (float[], float, bool) Step(int action)
        {
            // Simplified environment dynamics for illustration
            state[0] += action == 0 ? -0.1f : 0.1f;
            state[1] += action == 0 ? -0.1f : 0.1f;
            float reward = 1.0f;
            bool done = random.NextDouble() < 0.05;
            return (state, reward, done);
        }
    }
}
