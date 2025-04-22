import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math


class StableConvBlock(nn.Module):
    """A more stable convolutional block without batch normalization."""

    def __init__(self, in_channels, out_channels):
        super(StableConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class ActorNetwork(nn.Module):
    def __init__(self, action_dim=9):
        super(ActorNetwork, self).__init__()
        # Process local grid with CNN - simpler architecture
        self.conv1 = StableConvBlock(3, 16)
        self.conv2 = StableConvBlock(16, 32)

        # Process agent position, target position, and distance
        self.fc_pos = nn.Linear(5, 64)

        # Combine features
        self.fc_combine = nn.Linear(32 * 7 * 7 + 64, 128)  # Smaller layer
        self.fc_policy = nn.Linear(128, action_dim)

        # Initialize with small weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)  # Very small initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        # Extract observation components with safety checks
        if "local_grid" not in obs:
            raise ValueError("Missing 'local_grid' in observation")

        local_grid = obs["local_grid"]

        # Handle tensor shapes safely
        if len(local_grid.shape) == 3:  # Missing batch dimension
            local_grid = local_grid.unsqueeze(0)

        if len(local_grid.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor for local_grid, got shape {local_grid.shape}"
            )

        # Ensure the tensor is properly permuted
        local_grid = local_grid.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Position information
        agent_pos = obs["agent_pos"]
        target_pos = obs["target_pos"]
        target_distance = obs["target_distance"]

        # Forward pass with gradient checks
        with torch.set_grad_enabled(True):
            # Process grid using CNN
            x_grid = F.relu(self.conv1(local_grid))
            x_grid = F.dropout(x_grid, p=0.1, training=self.training)  # Add dropout
            x_grid = F.relu(self.conv2(x_grid))
            x_grid = F.dropout(x_grid, p=0.1, training=self.training)  # Add dropout

            # Safely flatten - ALWAYS use reshape instead of view
            x_grid = x_grid.reshape(x_grid.size(0), -1)

            # Process position info
            x_pos = torch.cat([agent_pos, target_pos, target_distance], dim=1)
            x_pos = F.relu(self.fc_pos(x_pos))

            # Safely combine
            x_combined = torch.cat([x_grid, x_pos], dim=1)
            x = F.relu(self.fc_combine(x_combined))
            x = F.dropout(x, p=0.1, training=self.training)  # Add dropout

            # Compute logits with clip for numerical stability
            logits = self.fc_policy(x)
            logits = torch.clamp(logits, -10.0, 10.0)  # Clip extreme values

            # Use numerically stable softmax
            action_probs = F.softmax(logits, dim=1)

            # Final safety check
            if torch.isnan(action_probs).any():
                print("WARNING: NaN detected in actor outputs")
                action_probs = torch.ones_like(action_probs) / action_probs.shape[1]

        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, num_agents=8):
        super(CriticNetwork, self).__init__()
        # Simpler architecture
        self.conv1 = StableConvBlock(3, 16)
        self.conv2 = StableConvBlock(16, 32)

        # Process agent position, target position, and distance
        self.fc_pos = nn.Linear(5, 64)

        # Combine features - smaller network
        self.fc_combine = nn.Linear(32 * 7 * 7 + 64, 128)
        self.fc_value = nn.Linear(128, 1)

        # Initialize with small weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)  # Very small initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        # Safe handling similar to ActorNetwork
        local_grid = obs["local_grid"]

        # Handle tensor shapes safely
        if len(local_grid.shape) == 3:
            local_grid = local_grid.unsqueeze(0)

        local_grid = local_grid.permute(0, 3, 1, 2).contiguous()

        agent_pos = obs["agent_pos"]
        target_pos = obs["target_pos"]
        target_distance = obs["target_distance"]

        # Forward pass with gradient checks
        with torch.set_grad_enabled(True):
            x_grid = F.relu(self.conv1(local_grid))
            x_grid = F.dropout(x_grid, p=0.1, training=self.training)
            x_grid = F.relu(self.conv2(x_grid))
            x_grid = F.dropout(x_grid, p=0.1, training=self.training)

            # Safely flatten - ALWAYS use reshape instead of view
            x_grid = x_grid.reshape(x_grid.size(0), -1)

            x_pos = torch.cat([agent_pos, target_pos, target_distance], dim=1)
            x_pos = F.relu(self.fc_pos(x_pos))

            x_combined = torch.cat([x_grid, x_pos], dim=1)
            x = F.relu(self.fc_combine(x_combined))

            value = self.fc_value(x)
            value = torch.clamp(value, -100.0, 100.0)  # Prevent extreme values

        return value


class MAPPOAgent:
    def __init__(
        self,
        num_agents=8,
        action_dim=9,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01,
        device="cpu",
        max_grad_norm=0.1,  # Much stricter gradient clipping
    ):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Track NaN occurrences
        self.nan_count = 0
        self.update_count = 0

        # Create networks
        self.create_networks()

        # Use much lower learning rates
        self.lr_actor = lr_actor * 0.1  # Reduce by factor of 10
        self.lr_critic = lr_critic * 0.1

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.lr_actor,
            weight_decay=1e-5,  # L2 regularization
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            weight_decay=1e-5,  # L2 regularization
        )

        # Create learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.95
        )

        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=100, gamma=0.95
        )

        # Memory for PPO
        self.reset_memory()

    def create_networks(self):
        """Create new actor and critic networks."""
        self.actor = ActorNetwork(self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.num_agents).to(self.device)

    def reset_memory(self):
        """Reset the agent's memory."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def preprocess_observation(self, obs):
        """Convert observation dict to tensors with safety checks."""
        processed = {}

        # Process observation components safely
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Check for NaN or infinite values
                if not np.isfinite(value).all():
                    print(f"WARNING: Non-finite values in observation {key}")
                    # Replace with zeros
                    value = np.zeros_like(value)

                processed[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            elif isinstance(value, dict):
                processed[key] = {
                    k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    for k, v in value.items()
                }

        return processed

    def select_action(self, obs, evaluate=False):
        """Select action based on current policy with extensive error checking."""
        # Process observation
        try:
            processed_obs = self.preprocess_observation(obs)
        except Exception as e:
            print(f"Error preprocessing observation: {e}")
            # Fall back to random action
            return random.randint(0, self.action_dim - 1)

        # Get action probability distribution
        try:
            with torch.no_grad():
                action_probs = self.actor(processed_obs)
                value = self.critic(processed_obs)

                # Check for NaN values
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    self.nan_count += 1
                    # If we've seen too many NaNs, reinitialize the networks
                    if self.nan_count > 10:
                        print("Too many NaNs detected, reinitializing networks")
                        self.create_networks()
                        self.nan_count = 0

                    # Use uniform distribution as fallback
                    action_probs = (
                        torch.ones((1, self.action_dim), device=self.device)
                        / self.action_dim
                    )
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return random action as fallback
            return random.randint(0, self.action_dim - 1)

        # Sample action
        try:
            if evaluate:
                action = torch.argmax(action_probs, dim=1).cpu().numpy()[0]
            else:
                # Create distribution safely
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().cpu().numpy()[0]

                # Get log probability for selected action
                log_prob = dist.log_prob(torch.tensor(action, device=self.device))

                # Store for PPO update
                self.log_probs.append(log_prob)
                self.values.append(value)
        except Exception as e:
            print(f"Error sampling action: {e}")
            action = random.randint(0, self.action_dim - 1)

            # Create dummy values for PPO
            dummy_log_prob = torch.tensor([-1.0], device=self.device)
            self.log_probs.append(dummy_log_prob)
            self.values.append(torch.tensor([0.0], device=self.device))

        return action

    def store_transition(self, state, action, reward, done):
        """Store transition with safety checks."""
        # Check if reward is finite
        if not np.isfinite(reward):
            print(f"WARNING: Non-finite reward detected: {reward}, clamping")
            reward = np.clip(reward, -10.0, 10.0)  # Prevent extreme rewards

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, next_value=None):
        """Update policy and value function with comprehensive error handling."""
        self.update_count += 1

        # Skip update if empty memory or too few samples
        if len(self.states) < 2:
            print("Skipping update: Not enough transitions stored")
            self.reset_memory()
            return

        try:
            # Calculate returns and advantages
            returns = []
            advantages = []
            gae = 0

            # Set terminal value
            if next_value is None:
                next_value = 0

            # Compute returns with GAE (with safety checks)
            for t in reversed(range(len(self.rewards))):
                # Calculate next return
                if t == len(self.rewards) - 1:
                    next_return = next_value
                    next_non_terminal = 1.0 - float(self.dones[t])
                else:
                    next_return = returns[0]
                    next_non_terminal = 1.0 - float(self.dones[t + 1])

                # Calculate discounted return
                current_return = (
                    self.rewards[t] + self.gamma * next_non_terminal * next_return
                )
                returns.insert(0, current_return)

                # Calculate advantage
                if t == len(self.rewards) - 1:
                    next_value_t = next_value
                else:
                    next_value_t = self.values[t + 1]

                delta = (
                    self.rewards[t]
                    + self.gamma * next_non_terminal * next_value_t
                    - self.values[t]
                )
                gae = delta + self.gamma * 0.95 * next_non_terminal * gae
                advantages.insert(0, gae)

            # Process batch of observations
            processed_states = {}

            try:
                # Stack observations carefully
                local_grids = np.stack([s["local_grid"] for s in self.states])
                processed_states["local_grid"] = torch.FloatTensor(local_grids).to(
                    self.device
                )

                processed_states["agent_pos"] = torch.FloatTensor(
                    np.vstack([s["agent_pos"] for s in self.states])
                ).to(self.device)

                processed_states["target_pos"] = torch.FloatTensor(
                    np.vstack([s["target_pos"] for s in self.states])
                ).to(self.device)

                processed_states["target_distance"] = torch.FloatTensor(
                    np.vstack([s["target_distance"] for s in self.states])
                ).to(self.device)
            except Exception as e:
                print(f"Error processing batch of states: {e}")
                self.reset_memory()
                return

            # Convert to tensors
            actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
            old_log_probs = torch.cat(self.log_probs).detach()
            returns = torch.tensor(returns, dtype=torch.float).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)

            # Normalize advantages safely
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # PPO update (multiple epochs)
            for epoch in range(5):  # Fewer epochs (was 10)
                # Forward pass
                action_probs = self.actor(processed_states)
                values = self.critic(processed_states).squeeze()

                # Check for NaN values
                if torch.isnan(action_probs).any() or torch.isnan(values).any():
                    print(f"WARNING: NaN detected during update epoch {epoch}")

                    # If we've hit too many NaNs, break early
                    if epoch > 0:  # Only break if we've completed at least one update
                        break

                    # Replace NaNs
                    action_probs = torch.nan_to_num(action_probs, nan=1e-5)
                    action_probs = F.softmax(torch.log(action_probs + 1e-10), dim=-1)
                    values = torch.nan_to_num(values, nan=0.0)

                # Create distribution and get new log probs
                try:
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                except Exception as e:
                    print(f"Error creating distribution: {e}")
                    break

                # Ensure tensor dimensions match
                if values.dim() == 0 and returns.dim() == 1:
                    values = values.unsqueeze(0)
                elif values.dim() == 1 and returns.dim() == 0:
                    returns = returns.unsqueeze(0)

                # Compute critic loss with shape checking
                if values.shape != returns.shape:
                    print(
                        f"Shape mismatch: values {values.shape}, returns {returns.shape}"
                    )
                    values = values.view(-1)
                    returns = returns.view(-1)

                    # Ensure same batch dimension
                    min_len = min(len(values), len(returns))
                    values = values[:min_len]
                    returns = returns[:min_len]

                critic_loss = F.mse_loss(values, returns)

                # Calculate ratios and PPO objectives
                try:
                    ratios = torch.exp(new_log_probs - old_log_probs)

                    # Clamp ratios for stability
                    ratios = torch.clamp(ratios, 0.1, 10.0)

                    surr1 = ratios * advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        * advantages
                    )

                    # Calculate actor loss (negative because we're maximizing)
                    actor_loss = -torch.min(surr1, surr2).mean()
                except Exception as e:
                    print(f"Error computing policy loss: {e}")
                    break

                # Compute total loss
                total_loss = (
                    actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                )

                # Check for NaN in loss
                if torch.isnan(total_loss):
                    print("NaN detected in loss, skipping update")
                    break

                # Perform update with careful gradient handling
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                try:
                    total_loss.backward()

                    # Very strict gradient clipping to prevent explosions
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )

                    # Check for NaN in gradients
                    has_nan_grad = False
                    for param in list(self.actor.parameters()) + list(
                        self.critic.parameters()
                    ):
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        print("NaN gradient detected, skipping optimizer step")
                    else:
                        self.actor_optimizer.step()
                        self.critic_optimizer.step()
                except Exception as e:
                    print(f"Error in backward/optimization step: {e}")
                    break

            # Step LR schedulers
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            # Monitor learning rates
            if self.update_count % 10 == 0:
                actor_lr = self.actor_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                print(
                    f"Current learning rates: actor={actor_lr:.6f}, critic={critic_lr:.6f}"
                )

        except Exception as e:
            print(f"Unexpected error during update: {e}")

        # Always clear memory after update
        self.reset_memory()
