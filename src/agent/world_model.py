"""
Transformer-based World Model for Delta-IRIS
Predicts next tokens, rewards, and done flags
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import logging
from .config import WorldModelConfig


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure mask has correct dtype if provided
        if mask is not None:
            mask = mask.to(dtype=x.dtype)
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class WorldModel(nn.Module):
    """Transformer-based world model for predicting environment dynamics"""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.action_embedding = nn.Embedding(config.action_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.sequence_length)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])
        
        # Output heads
        self.token_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.reward_head = nn.Linear(config.hidden_dim, 1)
        self.done_head = nn.Linear(config.hidden_dim, 1)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=self.token_embedding.weight.dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model
        
        Args:
            tokens: [batch_size, seq_len] - discrete tokens from tokenizer
            actions: [batch_size, seq_len] - actions taken
            
        Returns:
            Dictionary with predictions for next tokens, rewards, and dones
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Embed tokens and actions
        token_emb = self.token_embedding(tokens)  # [batch_size, seq_len, hidden_dim]
        action_emb = self.action_embedding(actions)  # [batch_size, seq_len, hidden_dim]
        
        # Combine embeddings (simple addition)
        x = token_emb + action_emb
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim] for pos encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask)
            
        # Apply final layer norm
        x = self.layer_norm(x)
        
        # Generate predictions
        next_token_logits = self.token_head(x)  # [batch_size, seq_len, vocab_size]
        reward_pred = self.reward_head(x).squeeze(-1)  # [batch_size, seq_len]
        done_pred = self.done_head(x).squeeze(-1)  # [batch_size, seq_len]
        
        return {
            'next_token_logits': next_token_logits,
            'reward_predictions': reward_pred,
            'done_predictions': done_pred,
            'hidden_states': x
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute world model losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Token prediction loss (cross-entropy)
        token_loss = F.cross_entropy(
            predictions['next_token_logits'].reshape(-1, self.config.vocab_size),
            targets['next_tokens'].reshape(-1),
            ignore_index=-1
        )
        
        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(
            predictions['reward_predictions'],
            targets['rewards']
        )
        
        # Done prediction loss (binary cross-entropy)
        done_loss = F.binary_cross_entropy_with_logits(
            predictions['done_predictions'],
            targets['dones'].float()
        )
        
        # Combined loss
        total_loss = token_loss + reward_loss + done_loss
        
        losses = {
            'token_loss': token_loss,
            'reward_loss': reward_loss, 
            'done_loss': done_loss,
            'total_world_model_loss': total_loss
        }
        
        return losses
        
    def generate(self, initial_tokens: torch.Tensor, initial_actions: torch.Tensor,
                 horizon: int, action_generator: callable) -> Dict[str, torch.Tensor]:
        """
        Generate rollouts using the world model
        
        Args:
            initial_tokens: [batch_size, context_len] - initial token sequence
            initial_actions: [batch_size, context_len] - initial action sequence  
            horizon: Number of steps to generate
            action_generator: Function that generates actions given current state
            
        Returns:
            Generated sequences
        """
        batch_size = initial_tokens.shape[0]
        device = initial_tokens.device
        
        # Initialize sequences
        tokens = initial_tokens.clone()
        actions = initial_actions.clone()
        rewards = []
        dones = []
        
        with torch.no_grad():
            for step in range(horizon):
                # Get predictions from current sequence
                predictions = self.forward(tokens, actions)
                
                # Sample next tokens
                next_token_probs = F.softmax(predictions['next_token_logits'][:, -1], dim=-1)
                next_tokens = torch.multinomial(next_token_probs, 1)  # [batch_size, 1]
                
                # Get reward and done predictions
                reward_pred = predictions['reward_predictions'][:, -1:] # [batch_size, 1]
                done_pred = torch.sigmoid(predictions['done_predictions'][:, -1:])  # [batch_size, 1]
                
                # Generate next actions using provided function
                current_state = {
                    'tokens': tokens,
                    'actions': actions,
                    'hidden_states': predictions['hidden_states'][:, -1:]
                }
                next_actions = action_generator(current_state)  # [batch_size, 1]
                
                # Append to sequences
                tokens = torch.cat([tokens, next_tokens], dim=1)
                actions = torch.cat([actions, next_actions], dim=1)
                rewards.append(reward_pred)
                dones.append(done_pred)
                
                # Stop if all environments are done
                if torch.all(done_pred > 0.5):
                    break
                    
        return {
            'generated_tokens': tokens,
            'generated_actions': actions,
            'predicted_rewards': torch.cat(rewards, dim=1) if rewards else torch.empty(batch_size, 0),
            'predicted_dones': torch.cat(dones, dim=1) if dones else torch.empty(batch_size, 0)
        }
        
    def _sanity_check(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]):
        """Perform sanity checks on world model predictions"""
        
        # Check for NaN/Inf values
        for key, tensor in predictions.items():
            if torch.isnan(tensor).any():
                self.logger.warning(f"NaN values in {key}")
            if torch.isinf(tensor).any():
                self.logger.warning(f"Inf values in {key}")
                
        # Check prediction ranges
        token_logits = predictions['next_token_logits']
        if token_logits.max() > 10 or token_logits.min() < -10:
            self.logger.warning(f"Token logits out of reasonable range: [{token_logits.min():.3f}, {token_logits.max():.3f}]")
            
        # Log accuracy metrics
        if hasattr(self, '_eval_step_count'):
            self._eval_step_count += 1
            if self._eval_step_count % 50 == 0:
                # Token prediction accuracy
                token_preds = torch.argmax(token_logits, dim=-1)
                token_acc = (token_preds == targets['next_tokens']).float().mean()
                self.logger.debug(f"Token prediction accuracy: {token_acc:.3f}")
        else:
            self._eval_step_count = 1
