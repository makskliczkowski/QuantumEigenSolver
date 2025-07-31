import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, List, Tuple, Union
from enum import Enum
from tqdm import trange

# ------------------------------------------------------------------------------------

from ent_loss_np import *

# ------------------------------------------------------------------------------------

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

# ---------------------------------------------------------------------------
"""
Ideas:
-   Brief: Try to maximize the purity of each subsystem - purity(rho_a) + purity(rho_b) = Tr(rho_a^2) + Tr(rho_b^2) to the reward
    This is a measure of how close the state is to being a pure state.
    Comment: ...
-   Brief: Energy-variance penalty: penalize states with high energy variance in the full Hamiltonian - if we are able
    to disentangle the state, the energy variance should be lower - no superpositions of degenerate states.
    Comment: ...
-   Brief: Orthogonality penalty: penalize states that are not orthogonal to other states in the training set.
    This encourages the agent to find states that are distinct from the training set.
    Comment: ...
-   Brief: Try to minimize the number of rotations needed to disentangle the state - quantum computing purpose is to
    minimize the number of gates needed to perform a task due to noise and decoherence.
    Comment: ...
-   Schmidt rank penalty: penalize states with high Schmidt rank - this is a measure of how entangled the state is.
    Comment: ...
"""

class ObjectiveType(Enum):
    """Enum for different objective types"""
    ENTANGLEMENT    = "entanglement"            # minimize entanglement entropy
    PURITY          = "purity"                  # maximize purity of subsystems
    NONGAUSSIANITY  = "non_gaussianity"         # maximize non-Gaussianity of the state
    ROTATIONS       = "rotations"               # minimize number of rotations, Pareto with entanglement
    SCHMIDT_RANK    = "schmidt_rank"            # minimize Schmidt rank - NOT IMPLEMENTED
    ENERGY_VAR      = "energy_variance"         # minimize energy variance - NOT IMPLEMENTED
    ORTHOGONALITY   = "orthogonality"           # maximize orthogonality - NOT IMPLEMENTED

@dataclass
class ObjectiveConfig:
    """Configuration for objective function"""
    entropy_weight          : float = 1.0
    purity_weight           : float = 0.0
    non_gaussianity_weight  : float = 0.0
    rotations_weight        : float = 0.0
    schmidt_weight          : float = 0.0       # NOT IMPLEMENTED
    energy_weight           : float = 0.0       # NOT IMPLEMENTED
    orthogonality_weight    : float = 0.0       # NOT IMPLEMENTED

# ---------------------------------------------------------------------------

class QuantumStateEnvironment:
    """Environment for quantum state disentanglement task"""
    
    def __init__(self, 
                    # state parameters
                    gamma                       : int,
                    dim_a                       : int,
                    dim_b                       : int,
                    train_states                : List[np.ndarray],
                    objective_config            : ObjectiveConfig           = ObjectiveConfig(),
                    org_states                  : List[np.ndarray]          = None,
                    # simulation parameters
                    rotation_angles             : Union[List[float], int]   = None,
                    rotation_angles_phi         : List[float]               = None,
                    unitary_agent               : bool                      = False,
                    # machine learning parameters
                    max_steps                   : int                       = 100,
                    target_entropy_threshold    : float                     = 1e-6,
                    # other
                    logger                      : Any                       = None,
                    iscomplex                   : bool                      = False
                ):
        
        self.logger         = logger
        self.iscomplex      = iscomplex
        self.dtype          = np.complex64 if iscomplex else np.float32
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if logger is not None:
            logger.info("I am a quantum state environment!", color="blue", lvl=1)

        # state related parameters
        self.org_states     = org_states
        self.state          = None
        self.states         = train_states if isinstance(train_states, list) else [train_states]
        self.dimension      = train_states[0].shape[0]
        self.dim_a          = dim_a
        self.dim_b          = dim_b
        self.gamma          = gamma
        
        # objective configuration
        self.obj_config     = objective_config
        if logger:
            for attr in ObjectiveConfig.__dataclass_fields__:
                if getattr(self.obj_config, attr) > 0:
                    logger.info(f"Objective config '{attr}': {getattr(self.obj_config, attr)}", color="red", lvl=2)

        # simulation parameters
        self.max_steps      = max_steps
        self.ent_threshold  = target_entropy_threshold
        self.unitary_agent  = unitary_agent
        
        # default rotation angles (in radians)
        self.rotation_angles, self.rotation_angles_phi = None, None
        self._initialize_angles(rotation_angles, rotation_angles_phi)
        
        # action space: (coeff1_idx, coeff2_idx, angle_idx, phi)
        self.n_coefficient_pairs    = (gamma * (gamma - 1)) // 2
        self.n_angles               = len(self.rotation_angles)
        self.n_angles_phi           = len(self.rotation_angles_phi)
        self.action_space_size      = self.n_coefficient_pairs * self.n_angles * self.n_angles_phi

        # callable cost function
        self.metrics                = None
        self.max_entropy            = np.log(self.dim_a)
    
        # reset environment
        self.reset()
        
    # -----------------------------------------------------------------------
    #! Private initialization methods
    # -----------------------------------------------------------------------

    def _initialize_angles(self, rotation_angles: Union[List[float], int], rotation_angles_phi: Union[List[float], int]) -> List[float]:
        # default rotation angles (in radians)
        if rotation_angles is None:
            self.rotation_angles    = [np.pi/8, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        elif isinstance(rotation_angles, int) or isinstance(rotation_angles, float):
            d_theta                 = np.pi / rotation_angles
            self.rotation_angles    = [d_theta * i for i in range(-rotation_angles, rotation_angles + 1)]
        elif isinstance(rotation_angles, list) or isinstance(rotation_angles, np.ndarray):
            self.rotation_angles    = rotation_angles
        self.rotation_angles_phi    = rotation_angles_phi if rotation_angles_phi is not None else [0.0]

    # -----------------------------------------------------------------------
    
    def reset(self):
        """Reset to uniform state"""
        
        if self.unitary_agent:
            # is unitary matrix consisting of gamma x gamma rotations
            self.state              = np.eye(self.gamma, dtype=self.dtype) 
        else:
            # is vector of coefficients for a single mixed state
            self.state              = np.ones(self.gamma, dtype=self.dtype) / np.sqrt(self.gamma)
            
        self.step_count             = 0
        self.metrics                = self._calculate_all_metrics(self.state)
        self.initial_entropy        = self.metrics['entropy']
        self.best_entropy           = self.initial_entropy
        return self._get_state_representation()
    
    # -----------------------------------------------------------------------
    #! Private methods for state representation and actions
    # -----------------------------------------------------------------------
    
    def _get_state_representation(self):
        """Convert complex state to real representation for neural network"""
        # Concatenate real and imaginary parts, plus magnitude and phase

        if not self.unitary_agent:
            # Single state representation
            real_part                   = np.real(self.state)
            imag_part                   = np.imag(self.state)
            magnitude                   = np.abs(self.state)
            phase                       = np.angle(self.state)
            current_metrics             = self._calculate_all_metrics(self.state)
            # Add step information and entropy history
            step_info                   = np.array([
                                                self.step_count / self.max_steps, 
                                                current_metrics['entropy'] / (self.max_entropy + 1e-8),
                                                current_metrics['purity'],
                                                current_metrics['nongaussianity']
                                            ])
            features                    = np.concatenate([
                                                real_part, 
                                                imag_part, 
                                                magnitude, 
                                                phase, 
                                                step_info
                                            ])
            return np.nan_to_num(features).astype(np.float32)
        else:
            # Unitary matrix representation - more efficient and stable
            real_part                   = np.real(self.state).flatten()
            imag_part                   = np.imag(self.state).flatten()
            
            # Unitarity measure
            unitarity_error             = np.linalg.norm(self.state.conj().T @ self.state - np.eye(self.gamma), 'fro')
            
            # Current metrics
            current_metrics             = self._calculate_all_metrics(self.state)

            # Combine features
            features                    = np.concatenate([
                                                real_part, 
                                                imag_part,
                                                np.array([
                                                    self.step_count / self.max_steps, 
                                                    current_metrics['entropy'] / (self.max_entropy + 1e-8),
                                                    current_metrics['purity'],
                                                    current_metrics['nongaussianity'],
                                                    unitarity_error
                                                ])
                                            ])
            return np.nan_to_num(features).astype(np.float32)
        
    # -----------------------------------------------------------------------
    
    def step(self, action: int):
        """Execute action and return (next_state, reward, done, info)"""
        
        if self.step_count >= self.max_steps:
            return self._get_state_representation(), 0.0, True, {}
        
        # Decode action - currently real rotations
        # Decode action
        pair_idx, angle_idx, phi_idx = self._decode_action(action)
        angle                   = self.rotation_angles[angle_idx]
        phi                     = self.rotation_angles_phi[phi_idx]
        coeff1_idx, coeff2_idx  = self._get_coefficient_pair(pair_idx)
        
        # Apply rotation
        old_metrics             = self._calculate_all_metrics(self.state)
        self._apply_rotation(coeff1_idx, coeff2_idx, angle, phi)
        new_metrics             = self._calculate_all_metrics(self.state)
        
        # Calculate reward
        entr_reduction          = old_metrics['entropy'] - new_metrics['entropy']
        done                    = (self.step_count >= self.max_steps or new_metrics['entropy'] < self.ent_threshold)
        reward                  = self._calculate_reward(old_metrics, new_metrics, done)

        # Update tracking
        self.step_count        += 1
        if new_metrics['entropy'] < self.best_entropy:
            self.best_entropy = new_metrics['entropy']
        
        # Check termination
        done                    = (self.step_count >= self.max_steps or new_metrics['entropy'] < self.ent_threshold)

        info = {
            'entropy'           : new_metrics['entropy'],
            'entropy_reduction' : entr_reduction,
            'purity'            : new_metrics['purity'],
            'nongaussianity'    : new_metrics['nongaussianity'],
            'best_entropy'      : self.best_entropy
        }
        
        return self._get_state_representation(), reward, done, info
    
    # -----------------------------------------------------------------------
    #! Private methods for action decoding and application
    # -----------------------------------------------------------------------

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decode action index to components for unitary agent"""
        phi_idx         = action % self.n_angles_phi
        action        //= self.n_angles_phi
        angle_idx       = action % self.n_angles
        action        //= self.n_angles
        pair_idx        = action % self.n_coefficient_pairs
        return pair_idx, angle_idx, phi_idx

    # -----------------------------------------------------------------------

    def _get_coefficient_pair(self, pair_idx: int) -> Tuple[int, int]:
        """Get coefficient indices from pair index"""
        # Convert linear index to upper triangular matrix indices
        k = 0
        for i in range(self.gamma):
            for j in range(i + 1, self.gamma):
                if k == pair_idx:
                    return i, j
                k += 1
        raise ValueError(f"Invalid pair index: {pair_idx}")

    # -----------------------------------------------------------------------

    def _apply_rotation(self, idx1: int, idx2: int, angle: float, phi: float = 0.0):
        """Apply rotation to two coefficients"""
        
        c, s            = np.cos(angle), np.sin(angle)
        if self.unitary_agent:
            rotation_matrix = np.eye(self.gamma, dtype=self.dtype)
            if self.iscomplex:
                rotation_matrix[idx1, idx1] = c
                rotation_matrix[idx1, idx2] = -s * np.exp(1j * phi)
                rotation_matrix[idx2, idx1] = s * np.exp(-1j * phi)
                rotation_matrix[idx2, idx2] = c
            else:
                rotation_matrix[idx1, idx1] = c
                rotation_matrix[idx1, idx2] = -s
                rotation_matrix[idx2, idx1] = s
                rotation_matrix[idx2, idx2] = c
            self.state  = rotation_matrix @ self.state
        else:
            if self.iscomplex:
                rotation         = np.exp(1j * phi)
                self.state[idx1] = c * self.state[idx1] - s * rotation * self.state[idx2]
                self.state[idx2] = s * np.conj(rotation) * self.state[idx1] + c * self.state[idx2]
            else:
                self.state[idx1] = c * self.state[idx1] - s * self.state[idx2]
                self.state[idx2] = s * self.state[idx1] + c * self.state[idx2]
            self.state /= np.linalg.norm(self.state)

    # -----------------------------------------------------------------------
    #! Reward calculation and metrics
    # -----------------------------------------------------------------------
    
    def _calculate_reward(self, old_metrics: dict, new_metrics: dict, done: bool = False) -> float:
        """Calculate reward based on entropy change"""
        
        # 1. Small penalty for each step to encourage efficiency
        step_penalty = -0.1 * (1 - self.step_count / self.max_steps)

        # 2. Reward for incremental improvement (optional but can help)
        incremental_reward = 0
        if self.obj_config.entropy_weight > 0:
            entropy_reduction   = old_metrics['entropy'] - new_metrics['entropy']
            incremental_reward += self.obj_config.entropy_weight * (entropy_reduction / self.max_entropy)

        if self.obj_config.purity_weight > 0:
            purity_increase     = new_metrics['purity'] - old_metrics['purity']
            incremental_reward += self.obj_config.purity_weight * purity_increase * 5.0
            
        # 3. Large terminal reward based on the final entropy
        terminal_reward = 0
        if done:
            # Reward is inversely proportional to the final entropy.
            # A final entropy of 0 gets the max reward of 1000.
            # A final entropy equal to the max possible entropy gets 0 reward.
            frac = max(0, (self.ent_threshold - new_metrics['entropy']) / self.ent_threshold)
            terminal_reward = 1000.0 * (1 - new_metrics['entropy'] / self.max_entropy) + 500.0 * frac**2

        return step_penalty + incremental_reward + terminal_reward
    
    # -----------------------------------------------------------------------
    
    def _calculate_all_metrics(self, state_transform):
        
        entanglements       = []
        purities            = []
        nongaussianities    = []

        # Go through each state and calculate metrics
        for state in self.states:
            transformed_states  = state @ state_transform # Apply transformation to the state
            entanglements.append(loss_entanglement_states(transformed_states, self.dim_a, self.dim_b))

            if self.obj_config.purity_weight > 0:
                # Purity loss is inverted because we want to maximize purity
                purities.append(1.0 + loss_purity_states(transformed_states, self.dim_a, self.dim_b))
            else:
                purities.append(0.0)
                
            if self.obj_config.non_gaussianity_weight > 0:
                # We want to minimize non-Gaussianity (which is 0 for Gaussian states) so we give it a negative sign
                nongaussianities.append(-loss_nongaussianity_states(transformed_states, self.dim_a, self.dim_b))
            else:
                nongaussianities.append(0.0)

        return {
            'entropy'           : np.mean(entanglements),
            'purity'            : np.mean(purities),
            'nongaussianity'    : np.mean(nongaussianities)
        }
    
    # -----------------------------------------------------------------------

# ---------------------------------------------------------------------------
#! Advanced neural network architecture for disentanglement
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Multi-head attention for quantum state features"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_k        = d_model // n_heads

        # Linear layers for query, key, value projections
        # These layers project the input features into different subspaces for attention
        # self.n_heads is the number of attention heads
        # Each head will have its own set of projections
        self.w_q        = nn.Linear(d_model, d_model)
        self.w_k        = nn.Linear(d_model, d_model)
        self.w_v        = nn.Linear(d_model, d_model)
        # Output projection after attention
        # This layer combines the outputs of all attention heads
        self.w_o        = nn.Linear(d_model, d_model)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len     = x.size(0), x.size(1)
        
        # Self-attention
        q                       = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k                       = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v                       = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores                  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights            = F.softmax(scores, dim=-1)
        attn_weights            = self.dropout(attn_weights)
        
        # Attention output - combine heads
        attn_output             = torch.matmul(attn_weights, v)
        attn_output             = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output                  = self.w_o(attn_output)
        return self.layer_norm(x + self.dropout(output))

# ---------------------------------------------------------------------------
#! Advanced neural network architecture for disentanglement
# ---------------------------------------------------------------------------

class DisentanglementNetwork(nn.Module):
    """
    Advanced neural network for quantum state disentanglement
    It uses residual connections, attention mechanisms, and a shared backbone
    to effectively learn disentanglement strategies.
    The architecture is designed to handle complex quantum states and
    capture quantum correlations through attention.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, n_res: int = 2, n_block: int = 2):
        super().__init__()
        
        # Feature extraction with residual connections
        # It is sequentially applied to the input state
        # It consists of a linear layer, layer normalization, ReLU activation, and dropout
        # This block extracts features from the input state and prepares it for attention
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for quantum correlations
        # It applies multi-head attention to the extracted features
        # It captures complex relationships between different parts of the quantum state
        # The attention block allows the model to focus on relevant features
        # Architecture:
        # - Input: feature vector of size hidden_dim
        # - Output: attended feature vector of size hidden_dim
        # - n_heads: number of attention heads (default is 8)
        # - d_model: dimension of the model (hidden_dim)
        # - dropout: dropout rate (default is 0.1)
        self.attention = AttentionBlock(hidden_dim, n_heads=4)
        
        # Shared backbone with residual connections
        self.backbone = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(n_res)
        ])
        
        # Actor head (policy network)
        # It outputs action logits for the given state
        # The actor network predicts the action probabilities based on the attended features
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        # It outputs the estimated value of the given state
        # The critic network predicts the value of the state based on the attended features
        # It helps in calculating advantages for policy updates
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

    # -----------------------------------------------------------------------
    #! Forward pass through the network
    # -----------------------------------------------------------------------
    
    def forward(self, state):
        batch_size  = state.size(0)
        
        # Feature extraction
        x           = self.feature_extractor(state)
        
        # Attention mechanism (reshape for sequence processing)
        x_seq       = x.unsqueeze(1)  # Add sequence dimension
        x_attended  = self.attention(x_seq).squeeze(1)
        
        # Residual backbone
        for block in self.backbone:
            residual    = x_attended
            x_attended  = block(x_attended) + residual
            x_attended  = F.relu(x_attended)
        
        # Actor and critic outputs
        action_logits   = self.actor(x_attended)
        value           = self.critic(x_attended)
        
        return action_logits, value

# ---------------------------------------------------------------------------
#! Proximal Policy Optimization (PPO) Agent for Quantum Disentanglement
# ---------------------------------------------------------------------------

class PPOAgent:
    """Proximal Policy Optimization agent for quantum disentanglement"""
    
    def __init__(self,
            state_dim       : int,
            action_dim      : int,
            lr              : float = 3e-4,
            gamma           : float = 0.99,
            gae_lambda      : float = 0.95,
            clip_epsilon    : float = 0.2,
            value_coeff     : float = 0.5,
            entropy_coeff   : float = 0.01,
            max_grad_norm   : float = 0.5,
            device          : str   = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device         = device
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_epsilon   = clip_epsilon
        self.value_coeff    = value_coeff
        self.entropy_coeff  = entropy_coeff
        self.max_grad_norm  = max_grad_norm

        # Networks
        self.network        = DisentanglementNetwork(state_dim, action_dim, hidden_dim=256).to(device)
        self.optimizer      = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4, eps=1e-5)

        # Learning rate scheduler
        self.scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

        # Experience buffer
        self.buffer         : List[Experience] = []
        
        # Training metrics
        self.training_metrics = {
            'policy_loss'           : [],
            'value_loss'            : [],
            'entropy_loss'          : [],
            'total_loss'            : [],
            'explained_variance'    : []
        }
    
    # -----------------------------------------------------------------------
    #! Methods for agent interaction with environment
    # -----------------------------------------------------------------------
    
    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action using current policy"""
        state_tensor        = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            
        # Create action distribution - use softmax for discrete actions
        action_probs        = F.softmax(action_logits, dim=-1)
        
        if training:
            # Sample from distribution
            dist            = Categorical(action_probs)
            action          = dist.sample()
            log_prob        = dist.log_prob(action)
        else:
            # Take best action
            action          = torch.argmax(action_probs, dim=-1)
            log_prob        = torch.log(action_probs.gather(1, action.unsqueeze(1))).squeeze(1)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store experience in buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done, log_prob, value))
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages  = []
        gae         = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta       = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae         = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    # -----------------------------------------------------------------------
    #! Methods for training and saving/loading the agent
    # -----------------------------------------------------------------------
    
    def update(self, epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO"""
        
        if len(self.buffer) < batch_size:
            return
                
        # Prepare batch data
        states              = torch.FloatTensor(np.array([exp.state for exp in self.buffer])).to(self.device)
        actions             = torch.LongTensor(np.array([exp.action for exp in self.buffer])).to(self.device)
        old_log_probs       = torch.FloatTensor(np.array([exp.log_prob for exp in self.buffer])).to(self.device)
        rewards             = [exp.reward for exp in self.buffer]
        values              = [exp.value for exp in self.buffer]
        dones               = [exp.done for exp in self.buffer]
        
        # Compute next values for GAE
        next_values         = values[1:] + [0.0] # Assume terminal value is 0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages          = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns             = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages          = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(epochs):
            # Shuffle data
            indices         = torch.randperm(len(self.buffer))
            
            # Batch training
            for start in range(0, len(self.buffer), batch_size):
                end                 = start + batch_size
                batch_indices       = indices[start:end]

                batch_states        = states[batch_indices]
                batch_actions       = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages    = advantages[batch_indices]
                batch_returns       = returns[batch_indices]
                
                # Forward pass
                action_logits, current_values   = self.network(batch_states)
                action_probs                    = F.softmax(action_logits, dim=-1)
                dist                            = Categorical(action_probs)
                
                # Compute ratios and losses
                current_log_probs   = dist.log_prob(batch_actions)
                ratio               = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Policy loss (clipped)
                surr1               = ratio * batch_advantages
                surr2               = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss         = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss          = F.mse_loss(current_values.squeeze(), batch_returns)
                
                # Entropy loss (encourage exploration)
                entropy_loss        = -dist.entropy().mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_coeff * value_loss + 
                            self.entropy_coeff * entropy_loss)
                
                # Backward pass
                # 1) Zero gradients
                self.optimizer.zero_grad()
                # 2) Compute gradients
                total_loss.backward()
                # 3) Clip gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                # 4) Update parameters
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
            'network_state_dict'    : self.network.state_dict(),
            'optimizer_state_dict'  : self.optimizer.state_dict(),
            'scheduler_state_dict'  : self.scheduler.state_dict(),
            'training_metrics'      : self.training_metrics
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = checkpoint['training_metrics']

# ---------------------------------------------------------------------------
#! Main training function for disentanglement agent
# ---------------------------------------------------------------------------

def train_disentanglement_agent(
    states_list             : List[np.ndarray],
    gamma                   : int,
    la                      : int,
    lb                      : int,
    config                  : ObjectiveConfig = ObjectiveConfig(),
    unitary_agent           : bool = False,                             # whether to use unitary agent or not - unitary agent uses matrix rotations
    iscomplex               : bool = False,                             # whether the states are complex or real
    rotation_angles         : Union[List[float], int] = None,           # for real Givens rotations
    episodes                : int = 5000,                               # number of training episodes for the agent
    max_steps_per_episode   : int = 100,                                # maximum steps per episode - the maximum number of rotations per episode
    update_frequency        : int = 100,                                # frequency of agent updates - how often to update the agent's policy
    save_frequency          : int = 500,                                # frequency of model saving - how often to save the model
    batch_size              : int = 64,                                 # batch size for training
    model_path              : str = 'pytorch/disentanglement_agent.pth',# path to save the model
    org_states              : np.ndarray = None,                        # original states for reference
    ent_threshold           : float = 1e-6,                             # target entropy threshold for termination
    logger                  : Any = None,                            # logger for logging information
):
    """Train the disentanglement agent"""
    
    # Create environment
    env = QuantumStateEnvironment(gamma                     = gamma, 
                                dim_a                       = 2**la,
                                dim_b                       = 2**lb,
                                train_states                = states_list,
                                objective_config            = config,
                                org_states                  = org_states,
                                rotation_angles             = rotation_angles,
                                unitary_agent               = unitary_agent,
                                iscomplex                   = iscomplex,
                                logger                      = logger,
                                max_steps                   = max_steps_per_episode,
                                target_entropy_threshold    = ent_threshold)

    # Create agent
    state_dim           = len(env._get_state_representation())
    action_dim          = env.action_space_size
    agent               = PPOAgent(state_dim, action_dim)
    
    # Training loop
    episode_rewards     = []
    episode_entropies   = []
    best_entropy        = float('inf')
    logger.info(f"State dimension (representation): {state_dim}, Action dimension: {action_dim}", color="blue", lvl=1)
    
    for episode in trange(episodes, desc="Training Episodes"):
        state           = env.reset()
        total_reward    = 0
        done            = False
        
        while not done:
            # Select action
            action, log_prob, value = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.store_experience(state, action, reward, next_state, done, log_prob, value)
            
            # Update state and reward
            state           = next_state
            total_reward   += reward
            
        # Log episode info
        if logger:
            logger.info(f"Episode {episode}: Total Reward: {total_reward:.4f}, "
                        f"Entropy: {info['entropy']:.6f}, Purity: {info['purity']:.4f}, "
                        f"Non-Gaussianity: {info['nongaussianity']:.4f}",
                        color="blue", lvl=2)
        
        # Update agent
        if episode % update_frequency == 0 and episode > 0:
            agent.update(epochs=4, batch_size=batch_size)
        
        # Track metrics
        final_entropy = info['entropy']
        episode_rewards.append(total_reward)
        episode_entropies.append(final_entropy)
        
        if final_entropy < best_entropy:
            best_entropy = final_entropy
        
        # Logging
        if episode % 100 == 0:
            avg_reward      = np.mean(episode_rewards[-100:])
            avg_entropy     = np.mean(episode_entropies[-100:])
            
            logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.4f}, "
                        f"Avg Entropy: {avg_entropy:.6f}, Purity: {info['purity']:.4f}, Non-Gaussianity: {info['nongaussianity']:.4f}, "
                        f"Std Entropy: {np.std(episode_entropies[-100:]):.6f}, "
                        f"Best Entropy: {best_entropy:.6f}",
                        color="green", lvl=2)
        
        # Save model
        if episode % save_frequency == 0 and episode > 0:
            agent.save(f"{model_path}_episode_{episode}.pth")
    
    # Final save
    agent.save(model_path)
    logger.info(f"Training completed! Best entropy achieved: {best_entropy:.6f}", color="green", lvl=1)
    
    return agent, episode_rewards, episode_entropies


# # Example usage
if __name__ == "__main__":
    from QES.general_python.common import flog, plot
    from ent_read_states import load_quantum_states, parse_arguments
    
    args        = parse_arguments()
    logger      = flog.get_global_logger()
    
    (org_states, org_entropies), (mix_states_real, mix_entropies_real), system_params = load_quantum_states(
        L               = args.L,
        gamma           = args.gamma,
        r               = args.r,
        n_states        = args.n_states,
        data_dir        = args.data_dir,
        mixture_index   = args.mixture_index,
        logger          = logger
    )
    
    la          = args.L // 2
    lb          = args.L // 2
    savedir     = f"{args.sav_dir}/pytorch"
    logger.info(f"Saving models to {savedir}", color="blue", lvl=1)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    config     = ObjectiveConfig(
        entropy_weight          = args.ent_weight,
        purity_weight           = args.pur_weight,
        non_gaussianity_weight  = args.non_weight,
        rotations_weight        = args.rot_weight
    )

    # Train the disentanglement agent
    logger.info("Starting training of disentanglement agent...", color="green", lvl=1)                
    agent, rewards, entropies = train_disentanglement_agent(
        states_list             = mix_states_real,
        gamma                   = int(args.gamma),
        la                      = la,
        lb                      = lb,
        rotation_angles         = args.k,
        episodes                = int(args.n_steps),
        max_steps_per_episode   = int(args.max_steps),
        update_frequency        = int(args.upd_freq),
        save_frequency          = int(args.sav_freq),
        batch_size              = int(args.batch_size),
        model_path              = f'{args.sav_dir}/pytorch/disentanglement_agent.pth',
        org_states              = org_states,
        ent_threshold           = args.ent_thr,
        logger                  = logger,
        unitary_agent           = args.unitary,
        iscomplex               = args.is_complex
    )
    
    fig, ax = plot.Plotter.get_subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    x       = plot.Plotter.plot(ax[0], x=np.arange(len(rewards)), y=rewards, color="blue")
    y       = plot.Plotter.plot(ax[1], x=np.arange(len(entropies)), y=entropies, color="orange")
    ax[0].set_ylabel("Reward")
    ax[1].set_ylabel(r"$\bar{S}$")
    ax[1].set_xlabel("Episodes")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")

    logger.info("Training completed successfully!", color="green", lvl=1)
    logger.info(f"Final rewards: {rewards[-1]}, Final entropies: {entropies[-1]}", color="blue", lvl=2)
    plt.show()