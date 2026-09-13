import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from aegisflow.config import TrainConfig
from aegisflow.rl.networks import QNetwork
from aegisflow.rl.replay import ReplayBuffer


class DoubleDQNAgent:

    def __init__(
        self,
        config,
        state_dim=None,
    ):

        self.config = config
        self.state_dim = state_dim

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -------------------------------------------------
        # Networks
        # -------------------------------------------------

        self.online_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Target network initially identical
        # to online network.

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.target_net.eval()

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.lr,
        )

        # -------------------------------------------------
        # Replay buffer
        # -------------------------------------------------

        self.replay = ReplayBuffer(
            config.replay_capacity
        )

        # -------------------------------------------------
        # Exploration
        # -------------------------------------------------

        self.epsilon = config.epsilon_start

        self.total_steps = 0
        self.gradient_steps = 0

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def select_action(
        self,
        state,
        training=True,
    ):

        # Epsilon-greedy exploration

        if training and random.random() < self.epsilon:

            return random.randrange(
                self.config.action_dim
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.online_net(
                state_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )

    # =====================================================
    # STORE EXPERIENCE
    # =====================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.replay.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # =====================================================
    # DOUBLE DQN TRAINING STEP
    # =====================================================

    def train_step(self):

        if len(self.replay) < self.config.min_replay:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(
            self.config.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # -------------------------------------------------
        # Current Q(s,a)
        # -------------------------------------------------

        current_q = self.online_net(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # -------------------------------------------------
        # Double DQN target
        # -------------------------------------------------

        with torch.no_grad():

            # Step 1:
            # ONLINE network selects the action.

            next_q_online = self.online_net(
                next_states
            )

            next_actions = torch.argmax(
                next_q_online,
                dim=1,
                keepdim=True,
            )

            # Step 2:
            # TARGET network evaluates that action.

            next_q_target = self.target_net(
                next_states
            )

            next_q = next_q_target.gather(
                1,
                next_actions,
            ).squeeze(1)

            target_q = (
                rewards
                + self.config.gamma
                * (1.0 - dones)
                * next_q
            )

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        # -------------------------------------------------
        # Gradient update
        # -------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent unstable gradients.

        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.gradient_steps += 1

        return float(loss.item())

    # =====================================================
    # TARGET NETWORK UPDATE
    # =====================================================

    def update_target(self):

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

    # =====================================================
    # EPSILON DECAY
    # =====================================================

    def decay_epsilon(self):

        decay = self.config.epsilon_decay

        self.epsilon = (
            self.config.epsilon_end
            + (
                self.epsilon
                - self.config.epsilon_end
            )
            * np.exp(-1.0 / decay)
        )

        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon,
        )