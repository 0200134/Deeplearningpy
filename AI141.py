import gym
import numpy as np

env = gym.make('CartPole-v1')

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        self.Q[state][action] += self.learning_rate * (reward + self.discount_rate * np.max(self.Q[next_state]) - self.Q[state][action])
        self.epsilon *= self.epsilon_decay

# Train the agent
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

# Test the agent
total_reward = 0
for _ in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
print("Average reward:", total_reward / 100)
