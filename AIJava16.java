import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.agent.legacy.Agent;
import org.deeplearning4j.rl4j.agent.legacy.AgentFactory;
import org.deeplearning4j.rl4j.environment.GameEnvironment;
import org.deeplearning4j.rl4j.learning.configuration.DQNConfiguration;

public class ReinforcementLearningAgent {

    public static void main(String[] args) throws Exception {
        int seed = 123;
        DataManager manager = new DataManager(true);
        Environment<Integer, Integer, DiscreteSpace> environment = new GameEnvironment(seed);
        DQNPolicy<Integer> policy = new DQNPolicy<>(DQNFactoryStdDense.builder().build(environment.getObservationSpace().size(), environment.getActionSpace().getSize()));
        Agent<Integer, Integer, DiscreteSpace> agent = new Agent<>(environment, policy);
        agent.train();

        // Save the policy for later use
        policy.save("trained_policy");
    }
}
