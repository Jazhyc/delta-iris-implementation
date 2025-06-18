"""
Delta-IRIS World Model with Transformer architecture, KV caching, and block attention
Based on the reference implementation
"""
from dataclasses import dataclass

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import FrameCnnConfig, FrameEncoder
from .slicer import Head
from .transformer import TransformerEncoder, TransformerConfig
from utils import LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_latents: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


@dataclass
class WorldModelConfig:
    latent_vocab_size: int
    num_actions: int
    image_channels: int
    image_size: int
    latents_weight: float
    rewards_weight: float
    ends_weight: float
    two_hot_rews: bool
    transformer_config: TransformerConfig
    frame_cnn_config: FrameCnnConfig


def init_weights(module: nn.Module) -> None:
    """Initialize model weights following Delta-IRIS convention"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transformation"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exp transformation (inverse of symlog)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    """Two-hot encoding for continuous values"""
    x.clamp_(x_min, x_max - 1e-5)
    buckets = torch.linspace(x_min, x_max, num_buckets).to(x.device)
    k = torch.searchsorted(buckets, x) - 1
    values = torch.stack((buckets[k + 1] - x, x - buckets[k]), dim=-1) / (buckets[k + 1] - buckets[k]).unsqueeze(-1)  
    two_hots = torch.scatter(x.new_zeros(*x.size(), num_buckets), dim=-1, index=torch.stack((k, k + 1), dim=-1), src=values)
    return two_hots


def compute_softmax_over_buckets(logits: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    """Compute expected value from two-hot logits"""
    buckets = torch.linspace(x_min, x_max, num_buckets).to(logits.device)
    probs = F.softmax(logits, dim=-1)
    return probs @ buckets


class WorldModel(nn.Module):
    """Delta-IRIS Transformer-based World Model"""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = TransformerEncoder(config.transformer_config)

        # Validate CNN output dimensions match transformer input
        cnn_output_size = ((config.image_size // 2 ** sum(config.frame_cnn_config.down)) ** 2) * config.frame_cnn_config.latent_dim
        assert cnn_output_size == config.transformer_config.embed_dim, \
            f"CNN output size {cnn_output_size} doesn't match transformer embed_dim {config.transformer_config.embed_dim}"

        # Frame CNN for processing observations
        self.frame_cnn = nn.Sequential(
            FrameEncoder(config.frame_cnn_config), 
            Rearrange('b t c h w -> b t 1 (h w c)'), 
            nn.LayerNorm(config.transformer_config.embed_dim)
        )

        # Token embeddings
        self.act_emb = nn.Embedding(config.num_actions, config.transformer_config.embed_dim)
        self.latents_emb = nn.Embedding(config.latent_vocab_size, config.transformer_config.embed_dim)

        # Define block patterns for different token types
        act_pattern = torch.zeros(config.transformer_config.tokens_per_block)
        act_pattern[1] = 1  # Action tokens at position 1 in each block

        act_and_latents_but_last_pattern = torch.zeros(config.transformer_config.tokens_per_block) 
        act_and_latents_but_last_pattern[1:-1] = 1  # Action + latent tokens but not the last one

        # Output heads using slicer pattern
        self.head_latents = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_and_latents_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), 
                nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, config.latent_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), 
                nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, 255 if config.two_hot_rews else 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), 
                nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, 2)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, sequence: torch.FloatTensor, use_kv_cache: bool = False) -> WorldModelOutput:      
        prev_steps = self.transformer.keys_values.size if use_kv_cache else 0
        num_steps = sequence.size(1)

        # Pass through transformer
        outputs = self.transformer(sequence, use_kv_cache=use_kv_cache)

        # Generate predictions using slicer heads
        logits_latents = self.head_latents(outputs, num_steps, prev_steps)
        logits_rewards = self.head_rewards(outputs, num_steps, prev_steps)
        logits_ends = self.head_ends(outputs, num_steps, prev_steps)

        return WorldModelOutput(outputs, logits_latents, logits_rewards, logits_ends)

    def compute_loss(self, batch, tokenizer, **kwargs) -> tuple:
        """
        Compute world model loss following Delta-IRIS approach
        
        Args:
            batch: Batch of data with observations, actions, rewards, ends, mask_padding
            tokenizer: Tokenizer for encoding observations
            
        Returns:
            Tuple of (LossWithIntermediateLosses, metrics_dict)
        """
        # Ensure we don't have more than one episode end per sequence
        assert torch.all(batch.ends.sum(dim=1) <= 1)

        # Encode observations to latent tokens using tokenizer
        with torch.no_grad():
            latent_tokens = tokenizer(batch.observations[:, :-1], batch.actions[:, :-1], batch.observations[:, 1:]).tokens

        b, _, k = latent_tokens.size()

        # Create transformer input sequence
        frames_emb = self.frame_cnn(batch.observations)
        act_tokens_emb = self.act_emb(rearrange(batch.actions, 'b t -> b t 1'))
        latent_tokens_emb = self.latents_emb(torch.cat((latent_tokens, latent_tokens.new_zeros(b, 1, k)), dim=1))
        
        # Interleave frame, action, and latent token embeddings
        sequence = rearrange(torch.cat((frames_emb, act_tokens_emb, latent_tokens_emb), dim=2), 'b t p1k e -> b (t p1k) e')
  
        # Forward pass through world model
        outputs = self(sequence)

        # Apply padding mask
        mask = batch.mask_padding

        # Latent token prediction loss
        labels_latents = latent_tokens[mask[:, :-1]].flatten()
        logits_latents = outputs.logits_latents[:, :-k][repeat(mask[:, :-1], 'b t -> b (t k)', k=k)]
        latent_acc = (logits_latents.max(dim=-1)[1] == labels_latents).float().mean()

        # Reward prediction loss with two-hot encoding if enabled
        if self.config.two_hot_rews:
            labels_rewards = two_hot(symlog(batch.rewards))
        else:
            labels_rewards = (batch.rewards.sign() + 1).long()

        # Compute losses
        loss_latents = F.cross_entropy(logits_latents, target=labels_latents) * self.config.latents_weight
        loss_rewards = F.cross_entropy(outputs.logits_rewards[mask], target=labels_rewards[mask]) * self.config.rewards_weight
        loss_ends = F.cross_entropy(outputs.logits_ends[mask], target=batch.ends[mask]) * self.config.ends_weight

        # Return losses and metrics
        losses = LossWithIntermediateLosses(loss_latents=loss_latents, loss_rewards=loss_rewards, loss_ends=loss_ends)
        metrics = {'latent_accuracy': latent_acc.item()}

        return losses, metrics

    @torch.no_grad()
    def burn_in(self, obs: torch.FloatTensor, act: torch.LongTensor, latent_tokens: torch.LongTensor, use_kv_cache: bool = False) -> torch.FloatTensor: 
        """
        Burn in the world model with a sequence of observations and actions
        
        Args:
            obs: Observations [batch_size, seq_len+1, ...]
            act: Actions [batch_size, seq_len, ...]
            latent_tokens: Latent tokens [batch_size, seq_len, num_tokens]
            use_kv_cache: Whether to use KV caching
            
        Returns:
            World model output sequence
        """
        assert obs.size(1) == act.size(1) + 1 == latent_tokens.size(1) + 1

        # Encode inputs
        x_emb = self.frame_cnn(obs)
        act_emb = rearrange(self.act_emb(act), 'b t e -> b t 1 e')
        q_emb = self.latents_emb(latent_tokens)
        
        # Create input sequence
        x_a_q = rearrange(torch.cat((x_emb[:, :-1], act_emb, q_emb), dim=2), 'b t k2 e -> b (t k2) e')
        wm_input_sequence = torch.cat((x_a_q, x_emb[:, -1]), dim=1)
        
        # Forward pass
        wm_output_sequence = self(wm_input_sequence, use_kv_cache=use_kv_cache).output_sequence

        return wm_output_sequence
