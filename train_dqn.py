import os
import sys
import math
import random
import collections
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env import SumoTrafficEnv

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

Transition = collections.namedtuple('Transition', 'state action reward next_state done')

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=Device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=Device).unsqueeze(-1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=Device).unsqueeze(-1)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=Device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=Device).unsqueeze(-1)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)


def train(num_episodes=50, max_steps=600, gui=False):
    env = SumoTrafficEnv(gui=gui, max_steps=max_steps)
    obs = env.reset()
    obs_size = len(obs)
    n_actions = 2

    policy_net = DQN(obs_size, n_actions).to(Device)
    target_net = DQN(obs_size, n_actions).to(Device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10000)

    gamma = 0.99
    batch_size = 64
    eps_start, eps_end, eps_decay = 1.0, 0.05, 2000
    steps_done = 0

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
            steps_done += 1
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32, device=Device).unsqueeze(0))
                    action = int(torch.argmax(q, dim=1).item())
            next_state, reward, done, _ = env.step_env(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target = rewards + gamma * max_next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if steps_done % 500 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        print(f"Episode {ep+1}/{num_episodes} reward={ep_reward:.2f}")
    env.close()
    torch.save(policy_net.state_dict(), 'dqn_sumo.pt')
    print('Training finished, model saved to dqn_sumo.pt')

if __name__ == '__main__':
    train(num_episodes=5, max_steps=300, gui=False)
