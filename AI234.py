import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import gym

class PPO:
    def __init__(self, state_dim, action_dim, action_bound, lr, gamma, clip_ratio, epochs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = optimizers.Adam(learning_rate=self.lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.lr)

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        dense = layers.Dense(64, activation='relu')(inputs)
        dense = layers.Dense(64, activation='relu')(dense)
        outputs = layers.Dense(self.action_dim, activation='tanh')(dense)
        outputs = outputs * self.action_bound  # Scale to action bounds
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        dense = layers.Dense(64, activation='relu')(inputs)
        dense = layers.Dense(64, activation='relu')(dense)
        outputs = layers.Dense(1)(dense)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = tf.squeeze(self.actor(state)).numpy()
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def compute_loss(self, old_log_probs, advantages, log_probs):
        ratios = tf.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return loss

    def train(self, states, actions, rewards, next_states, dones, old_log_probs):
        discounted_sum = 0
        discounted_rewards = []
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_rewards.insert(0, discounted_sum)
        discounted_rewards = np.array(discounted_rewards)
        advantages = discounted_rewards - tf.squeeze(self.critic(states))

        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                log_probs = self.actor(states)
                log_probs = -tf.reduce_sum(tf.square((actions - log_probs) / 0.1), axis=1)
                actor_loss = self.compute_loss(old_log_probs, advantages, log_probs)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                critic_loss = tf.reduce_mean(tf.square(discounted_rewards - self.critic(states)))
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

def main():
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    ppo = PPO(state_dim, action_dim, action_bound, lr=0.0003, gamma=0.99, clip_ratio=0.2, epochs=10)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

        while not done:
            action = ppo.get_action(state)
            next_state, reward, done, _ = env.step(action)
            log_prob = -tf.reduce_sum(tf.square((action - ppo.actor(np.expand_dims(state, axis=0))) / 0.1))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob.numpy())

            state = next_state
            episode_reward += reward

            if done:
                print(f'Episode {episode + 1}, Reward: {episode_reward}')
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                old_log_probs = np.array(log_probs)
                ppo.train(states, actions, rewards, next_states, dones, old_log_probs)
                break

if __name__ == '__main__':
    main()
