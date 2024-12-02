using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace PPOContinuousAction
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initialize environment and PPO agent
            var env = new ContinuousActionEnvironment();
            var agent = new PPOAgent(env.StateDim, env.ActionDim, env.ActionBound, lr: 0.0003, gamma: 0.99, clipRatio: 0.2, epochs: 10);

            // Training parameters
            int episodes = 1000;
            int batchSize = 64;

            for (int episode = 0; episode < episodes; episode++)
            {
                var state = env.Reset();
                double episodeReward = 0;
                var states = new List<NDArray>();
                var actions = new List<NDArray>();
                var rewards = new List<double>();
                var dones = new List<bool>();

                while (true)
                {
                    var action = agent.GetAction(state);
                    var (nextState, reward, done) = env.Step(action);

                    states.Add(state);
                    actions.Add(action);
                    rewards.Add(reward);
                    dones.Add(done);

                    state = nextState;
                    episodeReward += reward;

                    if (dones.Count >= batchSize)
                    {
                        agent.Train(states, actions, rewards, dones);
                        states.Clear();
                        actions.Clear();
                        rewards.Clear();
                        dones.Clear();
                    }

                    if (done)
                    {
                        Console.WriteLine($"Episode {episode + 1}, Reward: {episodeReward}");
                        break;
                    }
                }
            }
        }
    }

    public class PPOAgent
    {
        private readonly int stateDim;
        private readonly int actionDim;
        private readonly double actionBound;
        private readonly double lr;
        private readonly double gamma;
        private readonly double clipRatio;
        private readonly int epochs;

        private readonly Model actor;
        private readonly Model critic;
        private readonly Optimizer actorOptimizer;
        private readonly Optimizer criticOptimizer;

        public PPOAgent(int stateDim, int actionDim, double actionBound, double lr, double gamma, double clipRatio, int epochs)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.actionBound = actionBound;
            this.lr = lr;
            this.gamma = gamma;
            this.clipRatio = clipRatio;
            this.epochs = epochs;

            this.actor = BuildActor();
            this.critic = BuildCritic();
            this.actorOptimizer = tf.keras.optimizers.Adam(lr);
            this.criticOptimizer = tf.keras.optimizers.Adam(lr);
        }

        private Model BuildActor()
        {
            var inputs = keras.Input(shape: (stateDim));
            var dense1 = new Dense(64, activation: "relu").Apply(inputs);
            var dense2 = new Dense(64, activation: "relu").Apply(dense1);
            var outputs = new Dense(actionDim, activation: "tanh").Apply(dense2) * actionBound;
            var model = keras.Model(inputs, outputs);
            model.summary();
            return model;
        }

        private Model BuildCritic()
        {
            var inputs = keras.Input(shape: (stateDim));
            var dense1 = new Dense(64, activation: "relu").Apply(inputs);
            var dense2 = new Dense(64, activation: "relu").Apply(dense1);
            var outputs = new Dense(1).Apply(dense2);
            var model = keras.Model(inputs, outputs);
            model.summary();
            return model;
        }

        public NDArray GetAction(NDArray state, bool training = true)
        {
            var stateBatch = state.expand_dims(axis: 0);
            var action = actor.predict(stateBatch)[0];
            if (training)
            {
                var noise = np.random.normal(0, 0.1, action.shape);
                action += noise;
            }
            return np.clip(action, -actionBound, actionBound);
        }

        public void Train(List<NDArray> states, List<NDArray> actions, List<double> rewards, List<bool> dones)
        {
            var discountedRewards = ComputeDiscountedRewards(rewards, dones);
            var advantages = ComputeAdvantages(states, discountedRewards);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                using (var tape = tf.GradientTape())
                {
                    var stateBatch = np.array(states.ToArray());
                    var actionBatch = np.array(actions.ToArray());
                    var advantageBatch = np.array(advantages);

                    var logProbs = -tf.reduce_sum(tf.square((actionBatch - actor.predict(stateBatch)) / 0.1), axis: 1);
                    var oldLogProbs = -tf.reduce_sum(tf.square((actionBatch - actor.predict(stateBatch)) / 0.1), axis: 1);

                    var ratios = tf.exp(logProbs - oldLogProbs);
                    var surr1 = ratios * advantageBatch;
                    var surr2 = tf.clip_by_value(ratios, 1.0 - clipRatio, 1.0 + clipRatio) * advantageBatch;
                    var actorLoss = -tf.reduce_mean(tf.minimum(surr1, surr2));

                    var actorGradients = tape.gradient(actorLoss, actor.trainable_variables);
                    actorOptimizer.apply_gradients(zip(actorGradients, actor.trainable_variables));

                    var criticLoss = tf.reduce_mean(tf.square(discountedRewards - critic.predict(stateBatch)));
                    var criticGradients = tape.gradient(criticLoss, critic.trainable_variables);
                    criticOptimizer.apply_gradients(zip(criticGradients, critic.trainable_variables));
                }
            }
        }

        private NDArray ComputeDiscountedRewards(List<double> rewards, List<bool> dones)
        {
            double discountedSum = 0;
            var discountedRewards = new double[rewards.Count];
            for (int i = rewards.Count - 1; i >= 0; i--)
            {
                if (dones[i]) discountedSum = 0;
                discountedSum = rewards[i] + gamma * discountedSum;
                discountedRewards[i] = discountedSum;
            }
            return np.array(discountedRewards);
        }

        private NDArray ComputeAdvantages(List<NDArray> states, NDArray discountedRewards)
        {
            var advantages = discountedRewards - critic.predict(np.array(states.ToArray()));
            return advantages;
        }
    }

    public class ContinuousActionEnvironment
    {
        private Random rand = new Random();
        private float[] state = new float[4];

        public int StateDim => state.Length;
        public int ActionDim => 2;
        public float ActionBound => 1.0f;

        public NDArray Reset()
        {
            state = new float[] { (float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble() };
            return np.array(state);
        }

        public (NDArray, float, bool) Step(NDArray action)
        {
            state[0] += (float)action[0];
            state[1] += (float)action[1];
            state[2] += (float)action[0];
            state[3] += (float)action[1];

            float reward = 1.0f;
            bool done = rand.NextDouble() < 0.05;
            return (np.array(state), reward, done);
        }
    }
}
