import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any
import math

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class QuantumStateEnvironment:
    """Environment for quantum state disentanglement task"""
    
    def __init__(self, dimension: int, gamma_length: int, 
                 rotation_angles: List[float] = None,
                 max_steps: int = 100,
                 target_entropy_threshold: float = 1e-6):
        self.dimension = dimension
        self.gamma_length = gamma_length
        self.max_steps = max_steps
        self.target_entropy_threshold = target_entropy_threshold
        
        # Default rotation angles (in radians)
        if rotation_angles is None:
            self.rotation_angles = [np.pi/8, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        else:
            self.rotation_angles = rotation_angles
            
        # Action space: (coeff1_idx, coeff2_idx, angle_idx, is_complex)
        self.n_coefficient_pairs = (gamma_length * (gamma_length - 1)) // 2
        self.n_angles = len(self.rotation_angles)
        self.action_space_size = self.n_coefficient_pairs * self.n_angles * 2  # x2 for complex/real
        
        self.reset()
    
    def reset(self):
        """Reset to uniform state"""
        # Start with uniform normalized vector
        self.state = np.ones(self.gamma_length, dtype=complex) / np.sqrt(self.gamma_length)
        self.step_count = 0
        self.initial_entropy = self.calculate_entanglement_entropy(self.state)
        self.best_entropy = self.initial_entropy
        return self._get_state_representation()
    
    def _get_state_representation(self):
        """Convert complex state to real representation for neural network"""
        # Concatenate real and imaginary parts, plus magnitude and phase
        real_part = np.real(self.state)
        imag_part = np.imag(self.state)
        magnitude = np.abs(self.state)
        phase = np.angle(self.state)
        
        # Add step information and entropy history
        step_info = np.array([self.step_count / self.max_steps, 
                             self.best_entropy / (self.initial_entropy + 1e-8)])
        
        return np.concatenate([real_part, imag_part, magnitude, phase, step_info])
    
    def step(self, action: int):
        """Execute action and return (next_state, reward, done, info)"""
        if self.step_count >= self.max_steps:
            return self._get_state_representation(), 0.0, True, {}
        
        # Decode action
        pair_idx, angle_idx, is_complex = self._decode_action(action)
        coeff1_idx, coeff2_idx = self._get_coefficient_pair(pair_idx)
        angle = self.rotation_angles[angle_idx]
        
        # Apply rotation
        old_entropy = self.calculate_entanglement_entropy(self.state)
        self._apply_rotation(coeff1_idx, coeff2_idx, angle, is_complex)
        new_entropy = self.calculate_entanglement_entropy(self.state)
        
        # Calculate reward
        reward = self._calculate_reward(old_entropy, new_entropy)
        
        # Update tracking
        self.step_count += 1
        if new_entropy < self.best_entropy:
            self.best_entropy = new_entropy
        
        # Check termination
        done = (self.step_count >= self.max_steps or 
                new_entropy < self.target_entropy_threshold)
        
        info = {
            'entropy': new_entropy,
            'entropy_reduction': old_entropy - new_entropy,
            'best_entropy': self.best_entropy
        }
        
        return self._get_state_representation(), reward, done, info
    
    def _decode_action(self, action: int) -> Tuple[int, int, bool]:
        """Decode action index to components"""
        is_complex = action % 2
        action //= 2
        angle_idx = action % self.n_angles
        action //= self.n_angles
        pair_idx = action % self.n_coefficient_pairs
        return pair_idx, angle_idx, bool(is_complex)
    
    def _get_coefficient_pair(self, pair_idx: int) -> Tuple[int, int]:
        """Get coefficient indices from pair index"""
        # Convert linear index to upper triangular matrix indices
        k = 0
        for i in range(self.gamma_length):
            for j in range(i + 1, self.gamma_length):
                if k == pair_idx:
                    return i, j
                k += 1
        raise ValueError(f"Invalid pair index: {pair_idx}")
    
    def _apply_rotation(self, idx1: int, idx2: int, angle: float, is_complex: bool):
        """Apply rotation to two coefficients"""
        c1, c2 = self.state[idx1], self.state[idx2]
        
        if is_complex:
            # Complex rotation (multiply by e^(i*angle))
            rotation = np.exp(1j * angle)
            self.state[idx1] = c1 * np.cos(angle) - c2 * np.sin(angle) * rotation
            self.state[idx2] = c1 * np.sin(angle) * np.conj(rotation) + c2 * np.cos(angle)
        else:
            # Real rotation (Givens rotation)
            cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            self.state[idx1] = c1 * cos_angle - c2 * sin_angle
            self.state[idx2] = c1 * sin_angle + c2 * cos_angle
        
        # Renormalize to maintain unit norm
        self.state /= np.linalg.norm(self.state)
    
    def _calculate_reward(self, old_entropy: float, new_entropy: float) -> float:
        """Calculate reward based on entropy change"""
        entropy_reduction = old_entropy - new_entropy
        
        # Primary reward: entropy reduction
        reward = entropy_reduction * 100.0
        
        # Bonus for significant improvements
        if entropy_reduction > 0.01:
            reward += 10.0
        
        # Penalty for increasing entropy
        if entropy_reduction < 0:
            reward -= 5.0
        
        # Large bonus for reaching target
        if new_entropy < self.target_entropy_threshold:
            reward += 1000.0
        
        return reward
    
    def calculate_entanglement_entropy(self, state: np.ndarray) -> float:
        """Placeholder for entanglement entropy calculation - replace with your implementation"""
        # This is a placeholder - replace with your actual entanglement entropy calculation
        # For demonstration, using von Neumann entropy of reduced density matrix
        rho = np.outer(state, np.conj(state))
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        return -np.sum(eigenvals * np.log(eigenvals + 1e-12))


class AttentionBlock(nn.Module):
    """Multi-head attention for quantum state features"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Self-attention
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        output = self.w_o(attn_output)
        return self.layer_norm(x + self.dropout(output))


class DisentanglementNetwork(nn.Module):
    """Advanced neural network for quantum state disentanglement"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for quantum correlations
        self.attention = AttentionBlock(hidden_dim, n_heads=8)
        
        # Shared backbone with residual connections
        self.backbone = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(4)
        ])
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_residual_block(self, dim: int):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.5)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        batch_size = state.size(0)
        
        # Feature extraction
        x = self.feature_extractor(state)
        
        # Attention mechanism (reshape for sequence processing)
        x_seq = x.unsqueeze(1)  # Add sequence dimension
        x_attended = self.attention(x_seq).squeeze(1)
        
        # Residual backbone
        for block in self.backbone:
            residual = x_attended
            x_attended = block(x_attended) + residual
            x_attended = F.relu(x_attended)
        
        # Actor and critic outputs
        action_logits = self.actor(x_attended)
        value = self.critic(x_attended)
        
        return action_logits, value


class PPOAgent:
    """Proximal Policy Optimization agent for quantum disentanglement"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.network = DisentanglementNetwork(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=lr, 
            weight_decay=1e-4,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Experience buffer
        self.buffer = []
        
        # Training metrics
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'explained_variance': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        
        if training:
            # Sample from distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Take best action
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1))).squeeze(1)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store experience in buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done, log_prob, value))
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO"""
        if len(self.buffer) < batch_size:
            return
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in self.buffer]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in self.buffer]).to(self.device)
        rewards = [exp.reward for exp in self.buffer]
        values = [exp.value for exp in self.buffer]
        dones = [exp.done for exp in self.buffer]
        
        # Compute next values for GAE
        next_values = values[1:] + [0.0]  # Assume terminal value is 0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(self.buffer))
            
            for start in range(0, len(self.buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_logits, current_values = self.network(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                # Compute ratios and losses
                current_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Policy loss (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(current_values.squeeze(), batch_returns)
                
                # Entropy loss (encourage exploration)
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coeff * value_loss + 
                             self.entropy_coeff * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store metrics
                self.training_metrics['policy_loss'].append(policy_loss.item())
                self.training_metrics['value_loss'].append(value_loss.item())
                self.training_metrics['entropy_loss'].append(entropy_loss.item())
                self.training_metrics['total_loss'].append(total_loss.item())
        
        # Update learning rate
        self.scheduler.step()
        
        # Clear buffer
        self.buffer.clear()
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_metrics': self.training_metrics
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = checkpoint['training_metrics']


def train_disentanglement_agent(
    dimension: int,
    gamma_length: int,
    episodes: int = 5000,
    max_steps_per_episode: int = 100,
    update_frequency: int = 100,
    save_frequency: int = 500,
    model_path: str = 'disentanglement_agent.pth'
):
    """Train the disentanglement agent"""
    
    # Create environment
    env = QuantumStateEnvironment(dimension, gamma_length, max_steps=max_steps_per_episode)
    
    # Create agent
    state_dim = len(env._get_state_representation())
    action_dim = env.action_space_size
    agent = PPOAgent(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    episode_entropies = []
    best_entropy = float('inf')
    
    print(f"Starting training for {episodes} episodes...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, value = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done, log_prob, value)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        # Update agent
        if episode % update_frequency == 0 and episode > 0:
            agent.update()
        
        # Track metrics
        final_entropy = info['entropy']
        episode_rewards.append(total_reward)
        episode_entropies.append(final_entropy)
        
        if final_entropy < best_entropy:
            best_entropy = final_entropy
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_entropy = np.mean(episode_entropies[-100:])
            print(f"Episode {episode}: Avg Reward: {avg_reward:.4f}, "
                  f"Avg Entropy: {avg_entropy:.6f}, Best Entropy: {best_entropy:.6f}")
        
        # Save model
        if episode % save_frequency == 0 and episode > 0:
            agent.save(f"{model_path}_episode_{episode}.pth")
    
    # Final save
    agent.save(model_path)
    print(f"Training completed! Best entropy achieved: {best_entropy:.6f}")
    
    return agent, episode_rewards, episode_entropies


# Example usage
if __name__ == "__main__":
    # Train agent for 4D quantum state with 8-dimensional gamma vector
    agent, rewards, entropies = train_disentanglement_agent(
        dimension=4,
        gamma_length=8,
        episodes=2000,
        max_steps_per_episode=50
    )
    
    print("Training completed!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.4f}")
    print(f"Final average entropy: {np.mean(entropies[-100:]):.6f}")