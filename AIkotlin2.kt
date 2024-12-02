import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearningBuilder
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearningDiscrete
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.mdp.gym.GymEnv
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.IOException

object AdvancedDQN {
    @JvmStatic
    @Throws(IOException::class)
    fun main(args: Array<String>) {
        // Define the Q-learning configuration
        val qlConfiguration = QLearning.QLConfiguration(
            123,     // Random seed
            200,     // Max step by epoch
            150000,  // Max step
            20000,   // Max size of experience replay
            32,      // Batch size
            500,     // Target update (hard)
            10,      // Num step noop warmup
            0.01,    // Reward scaling
            0.99,    // Gamma
            1.0,     // TD error clipping
            0.1f,    // Min epsilon
            1000,    // Num step for epsilon anneal
            true     // Double DQN
        )

        // Define the neural network architecture
        val netConf = DQNFactoryStdDense.Configuration.builder()
            .updater(Adam(1e-2))
            .l2(0.001)
            .numHiddenNodes(64)
            .numLayer(3)
            .build()

        // Create the MDP environment (here using OpenAI Gym)
        val mdp: MDP = GymEnv<DiscreteSpace>("CartPole-v0", false, false)

        // Data manager to save training data
        val manager = DataManager(true)

        // Create the DQN agent
        val dql = QLearningDiscrete(mdp, netConf, qlConfiguration, manager)

        // Train the agent
        dql.train()

        // Save the trained model
        dql.policy.save("dqn_cartpole_model")

        // Close the MDP environment
        mdp.close()
    }
}
