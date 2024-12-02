import java.util.Random;

public class QLearningExample {
    // Environment dimensions
    private static final int STATES = 5;
    private static final int ACTIONS = 2;
    private static final double[][] REWARDS = {
            {0, 0}, {0, 0}, {0, 0}, {0, 1}, {1, 1}
    };

    // Q-learning parameters
    private static final double ALPHA = 0.1; // Learning rate
    private static final double GAMMA = 0.9; // Discount factor
    private static final double EPSILON = 0.1; // Exploration rate

    // Q-table
    private double[][] qTable = new double[STATES][ACTIONS];
    private Random random = new Random();

    public static void main(String[] args) {
        QLearningExample qLearning = new QLearningExample();
        qLearning.train(1000); // Train for 1000 episodes
        qLearning.displayQTable();
    }

    // Training the agent
    private void train(int episodes) {
        for (int episode = 0; episode < episodes; episode++) {
            int state = 0; // Start at initial state
            while (state < STATES - 1) {
                int action = chooseAction(state);
                int nextState = takeAction(state, action);
                double reward = REWARDS[state][action];
                updateQTable(state, action, nextState, reward);
                state = nextState;
            }
        }
    }

    // Choosing action based on epsilon-greedy policy
    private int chooseAction(int state) {
        if (random.nextDouble() < EPSILON) {
            return random.nextInt(ACTIONS); // Explore
        } else {
            return maxAction(state); // Exploit
        }
    }

    // Determining the best action from Q-table
    private int maxAction(int state) {
        if (qTable[state][0] > qTable[state][1]) {
            return 0;
        } else {
            return 1;
        }
    }

    // Simulating taking an action
    private int takeAction(int state, int action) {
        if (action == 1 && state < STATES - 1) {
            return state + 1;
        } else {
            return state;
        }
    }

    // Updating Q-table
    private void updateQTable(int state, int action, int nextState, double reward) {
        double oldQValue = qTable[state][action];
        double nextMaxQValue = Math.max(qTable[nextState][0], qTable[nextState][1]);
        qTable[state][action] = oldQValue + ALPHA * (reward + GAMMA * nextMaxQValue - oldQValue);
    }

    // Displaying Q-table
    private void displayQTable() {
        for (int state = 0; state < STATES; state++) {
            System.out.println("State " + state + ": " + qTable[state][0] + ", " + qTable[state][1]);
        }
    }
}
