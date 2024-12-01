import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQNAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRuleBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.learning.configuration.DQNConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

public class DoubleDQNExample {
    public static void main(String[] args) {
        int seed = 123;
        int numInputs = 4;
        int numOutputs = 2;
        int batchSize = 64;
        int maxEpisodes = 1000;

        // Define the environment
        Environment<Integer, Integer, DiscreteSpace> environment = new MyEnvironment();

        // Define the network configuration
        DQNConfiguration conf = DQNConfiguration.builder()
                .seed(seed)
                .maxEpochStep(2000)
                .batchSize(batchSize)
                .maxExperienceReplaySize(50000)
                .updateTarget(1000)
                .build();

        // Build the network
        DQN<Integer> dqnNetwork = DQN.builder().build(environment.getObservationSpace().size(), environment.getActionSpace().getSize());

        // Define the policy and learning algorithm
        Policy<Integer> policy = Policy.builder().network(dqnNetwork).build(environment.getActionSpace());
        DoubleDQNAlgorithm<Integer> algorithm = new DoubleDQNAlgorithm<>(dqnNetwork, conf);

        // Create and train the agent
        AgentLearner<Integer> agentLearner = AgentLearner.builder(environment, policy, algorithm).build();
        agentLearner.train();

        // Save the trained policy
        DataManager manager = new DataManager(true);
        manager.savePolicy("double_dqn_policy", policy);
    }
}
