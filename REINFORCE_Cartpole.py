"""
Prerequisites:

Python 3.9.16 or a compatible version should be installed on your system.
To install the required dependencies, run the following command:
pip install -r requirements.txt
"""

# Imports
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Normalize Discounted Rewards
def normalize(x):
    # Normalize the rewards by subtracting the mean and dividing by the standard deviation
    x -= torch.mean(x)
    x /= torch.std(x)
    return x

# Create Neural Network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        # Set up two fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # From inputs to hidden
        self.fc2 = nn.Linear(hidden_size, output_size)  # From hidden to outputs

    def forward(self, x):
        # Apply ReLU (Turn negitive numbers into 0) to the first fully connected layer
        x = torch.relu(self.fc1(x))
        # Calculate values of the hidden layer (with weights and biases)
        x = self.fc2(x)
        # Normalize the output of the last layer (hidden layer) into a probability distribution
        x = torch.softmax(x, dim=-1)
        return x

# Set up environment and neural network
env = gym.make("CartPole-v1")
input_size = env.observation_space.shape[0]  # Observation space dimension
hidden_size = 128  # Number of neurons in the hidden layer
output_size = env.action_space.n  # Number of actions

policy_network = NN(input_size, hidden_size, output_size)  # Construct the neural network
optimizer = optim.Adam(policy_network.parameters(), lr=0.003)  # Set up optimizer

done = False
num_episodes = 200  # Number of episodes to run
gamma = 0.99  # Discount factor for cumulative rewards
max_episode_steps = 1000

progress_bar = tqdm(total=num_episodes, desc = "Episodes", unit="episode", ncols=80) # Set up progress bar for episodes

cumulativeRewards = []  # Store cumulative rewards per episode
losses = [] # Store loss per episode
episode_lengths = [] # Store number of actions/steps per episode

for episode in range(num_episodes):

    # Option to render the last episode
    if episode == num_episodes - 1:
        user_input = input("\nWould you like to render the last episode? Press 'Enter' to render or 'skip' to continue without rendering.")

        if user_input.lower() == 'skip':
            print("Continuing without rendering.")
        else:
            print("Rendering the last episode...")
            env = gym.make("CartPole-v1", render_mode="human")  # Render the last episode
            ui_steps = input("Enter the number of steps for the last episode: ")
            max_episode_steps = ui_steps

    episode_rewards = []  # Store episode rewards
    episode_log_probs = []  # Store log probabilities of chosen actions
    steps = 0

    state, _ = env.reset()  # Reset the environment and get initial observation

    while True:
        
        #Converts state into tensor
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)

        action_probs = policy_network(state)  # Forward pass to get action probabilities

        action_dist = torch.distributions.Categorical(action_probs) # Creates a Categorical Distribution of action probabilites

        action = action_dist.sample()  # Sample an action from the distribution

        log_prob = action_dist.log_prob(action)  # Calculate the log probability of the chosen action

        next_state, reward, done, _, _ = env.step(action.item())  # Take action in the environment
        episode_rewards.append(reward)  # Store the reward
        episode_log_probs.append(log_prob)  # Store the log probability

        if done or steps > int(max_episode_steps):

            discounted_rewards = np.zeros_like(episode_rewards)  # Array to store discounted rewards
            cumulative_rewards = 0

            # Calculate cumulative rewards using discounting
            for t in reversed(range(len(episode_rewards))):
                # Update cumulative rewards by discounting previous cumulative rewards and adding the current reward
                cumulative_rewards = cumulative_rewards * gamma + episode_rewards[t]
                # Store the calculated cumulative rewards in the discounted_rewards array
                discounted_rewards[t] = cumulative_rewards


            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float) # Make discounted rewards into a tensor

            discounted_rewards = normalize(discounted_rewards) # Normalize the discounted rewards

            episode_log_probs = torch.stack(episode_log_probs)

            loss = torch.sum(-episode_log_probs * discounted_rewards) # Calculate the loss

            optimizer.zero_grad() # Clear gradients - reset the gradients of the optimizer

            loss.backward() # Backpropagate - calculate gradients of the loss with respect to model parameters

            optimizer.step() # Update parameters - adjust model parameters based on gradients using optimizer

            cumulativeRewards.append(cumulative_rewards)
            episode_lengths.append(steps)
            losses.append(loss.item())

            break

        state = next_state
        steps+=1

    progress_bar.update(1)

progress_bar.close()

env.close()

# Calculate rolling average of rewards
rewardsRollingAverage = np.convolve(cumulativeRewards, np.ones(25)/25, mode='full')
# Plot episode rewards
plt.plot(cumulativeRewards, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards.png")
plt.close()

# Plot losses
plt.plot(losses)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Loss Value")
plt.title("Episode Losses")
plt.savefig("episode_losses.png")
plt.close()

# Plot episode lengths
plt.plot(episode_lengths)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Amount of Steps/Actions")
plt.title("Episode Lengths")
plt.savefig("episode_lengths.png")
plt.close()