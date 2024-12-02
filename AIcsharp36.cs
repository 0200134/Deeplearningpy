using System;
using System.Collections.Generic;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Neuro.ActivationFunctions;
using Accord.Controls;
using Accord.MachineLearning;
using Accord.Math.Random;

namespace PPOImplementation
{
    class Program
    {
        static void Main(string[] args)
        {
            // Hyperparameters
            int stateDim = 4;  // Example state dimension (e.g., CartPole)
            int actionDim = 2; // Example action dimension (e.g., Left, Right)
            double actionBound = 1.0;
            double lr = 0.0003;
            double gamma = 0.99;
            double clipRatio = 0.2;
            int epochs = 10;

            // Initialize PPO agent
            PPOAgent ppo = new PPOAgent(stateDim, actionDim, actionBound, lr, gamma, clipRatio, epochs);

            // Environment simulation (example: CartPole)
            var env = new CartPoleEnvironment();
            int numEpisodes = 1000;

            for (int episode = 0; episode < numEpisodes; episode++)
            {
                var state = env.Reset();
                bool done = false;
                double episodeReward = 0;
                List<double[]> states = new List<double[]>();
                List<double[]> actions = new List<double[]>();
                List<double> rewards = new List<double>();
                List<double[]> nextStates = new List<double[]>();
                List<bool> dones = new List<bool>();

                while (!done)
                {
                    double[] action = ppo.GetAction(state);
                    var (nextState, reward, isDone) = env.Step(action);

                    states.Add(state);
                    actions.Add(action);
                    rewards.Add(reward);
                    nextStates.Add(nextState);
                    dones.Add(isDone);

                    state = nextState;
                    episodeReward += reward;

                    if (isDone)
                    {
                        Console.WriteLine($"Episode {episode + 1}, Reward: {episodeReward}");
                        ppo.Train(states, actions, rewards, nextStates, dones);
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

        private readonly ActivationNetwork actorNetwork;
        private readonly ActivationNetwork criticNetwork;

        public PPOAgent(int stateDim, int actionDim, double actionBound, double lr, double gamma, double clipRatio, int epochs)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.actionBound = actionBound;
            this.lr = lr;
            this.gamma = gamma;
            this.clipRatio = clipRatio;
            this.epochs = epochs;

            this.actorNetwork = BuildActor();
            this.criticNetwork = BuildCritic();
        }

        private ActivationNetwork BuildActor()
        {
            var network = new ActivationNetwork(new BipolarSigmoidFunction(), stateDim, 64, 64, actionDim);
            new GaussianWeights(network).Randomize();
            return network;
        }

        private ActivationNetwork BuildCritic()
        {
            var network = new ActivationNetwork(new BipolarSigmoidFunction(), stateDim, 64, 64, 1);
            new GaussianWeights(network).Randomize();
            return network;
        }

        public double[] GetAction(double[] state)
        {
            double[] action = actorNetwork.Compute(state);
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = Math.Max(Math.Min(action[i], actionBound), -actionBound);
            }
            return action;
        }

        public void Train(List<double[]> states, List<double[]> actions, List<double> rewards, List<double[]> nextStates, List<bool> dones)
        {
            double[] discountedRewards = ComputeDiscountedRewards(rewards, dones);
            double[] advantages = ComputeAdvantages(states, discountedRewards);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Train Actor Network
                var actorTeacher = new ResilientBackpropagationLearning(actorNetwork);
                for (int i = 0; i < states.Count; i++)
                {
                    double[] action = actorNetwork.Compute(states[i]);
                    double logProbOld = ComputeLogProbability(action, actions[i]);
                    double logProbNew = ComputeLogProbability(actorNetwork.Compute(states[i]), actions[i]);

                    double ratio = Math.Exp(logProbNew - logProbOld);
                    double surr1 = ratio * advantages[i];
                    double surr2 = Math.Max(Math.Min(ratio, 1 + clipRatio), 1 - clipRatio) * advantages[i];
                    double actorLoss = -Math.Min(surr1, surr2);

                    actorTeacher.Run(new double[][] { states[i] }, new double[][] { action });
                }

                // Train Critic Network
                var criticTeacher = new ResilientBackpropagationLearning(criticNetwork);
                for (int i = 0; i < states.Count; i++)
                {
                    double criticLoss = Math.Pow(discountedRewards[i] - criticNetwork.Compute(states[i])[0], 2);
                    criticTeacher.Run(new double[][] { states[i] }, new double[][] { new double[] { discountedRewards[i] } });
                }
            }
        }

        private double[] ComputeDiscountedRewards(List<double> rewards, List<bool> dones)
        {
            double[] discountedRewards = new double[rewards.Count];
            double discountedSum = 0;

            for (int i = rewards.Count - 1; i >= 0; i--)
            {
                if (dones[i])
                {
                    discountedSum = 0;
                }
                discountedSum = rewards[i] + gamma * discountedSum;
                discountedRewards[i] = discountedSum;
            }

            return discountedRewards;
        }

        private double[] ComputeAdvantages(List<double[]> states, double[] discountedRewards)
        {
            double[] advantages = new double[states.Count];
            for (int i = 0; i < states.Count; i++)
            {
                advantages[i] = discountedRewards[i] - criticNetwork.Compute(states[i])[0];
            }
            return advantages;
        }

        private double ComputeLogProbability(double[] predicted, double[] actual)
        {
            double logProb = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                logProb += -Math.Pow(actual[i] - predicted[i], 2) / (2 * 0.1 * 0.1);
            }
            return logProb;
        }
    }

    public class CartPoleEnvironment
    {
        private readonly Random random = new Random();
        private double[] state = new double[4];

        public double[] Reset()
        {
            state = new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble(), random.NextDouble() };
            return state;
        }

        public (double[], double, bool) Step(double[] action)
        {
            // Simplified environment dynamics for illustration
            state[0] += action[0];
            state[1] += action[1];
            state[2] += action[0];
            state[3] += action[1];
            double reward = 1.0;
            bool done = random.NextDouble() < 0.05;
            return (state, reward, done);
        }
    }
}
