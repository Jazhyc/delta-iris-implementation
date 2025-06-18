"""
Context-Aware Tokenizer Module
Delta encoding and context-aware tokenization for Delta-IRIS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class DeltaEncoder(nn.Module):
    """
    Delta (difference) encoder for context-aware tokenization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Delta encoding network
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Current + Previous
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """
        Encode delta between current and previous representations
        
        Args:
            current: [batch_size, ..., input_dim] current representation
            previous: [batch_size, ..., input_dim] previous representation
            
        Returns:
            delta_encoded: [batch_size, ..., output_dim] delta-encoded representation
        """
        # Concatenate current and previous
        combined = torch.cat([current, previous], dim=-1)
        
        # Apply delta encoding
        delta_encoded = self.delta_net(combined)
        
        # Add residual connection
        residual = self.residual_proj(current)
        delta_encoded = delta_encoded + residual
        
        return delta_encoded


class ContextMemory(nn.Module):
    """
    Memory module for maintaining context across sequences
    """
    
    def __init__(self, memory_size: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Memory slots
        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim))
        
        # Attention for reading from memory
        self.read_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Attention for writing to memory
        self.write_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Layer norms
        self.read_norm = nn.LayerNorm(embed_dim)
        self.write_norm = nn.LayerNorm(embed_dim)
        
        # Gate for memory updates
        self.update_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, query: torch.Tensor, update_memory: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from and optionally update memory
        
        Args:
            query: [batch_size, seq_len, embed_dim] query for memory
            update_memory: Whether to update memory with current query
            
        Returns:
            memory_output: [batch_size, seq_len, embed_dim] memory-augmented output
            updated_memory: [batch_size, memory_size, embed_dim] updated memory
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Expand memory for batch
        memory = self.memory.repeat(batch_size, 1, 1)
        
        # Read from memory
        memory_output, _ = self.read_attention(
            query.view(batch_size * seq_len, 1, embed_dim),
            memory.view(batch_size, 1, self.memory_size, embed_dim).repeat(1, seq_len, 1, 1).view(-1, self.memory_size, embed_dim),
            memory.view(batch_size, 1, self.memory_size, embed_dim).repeat(1, seq_len, 1, 1).view(-1, self.memory_size, embed_dim)
        )
        memory_output = memory_output.view(batch_size, seq_len, embed_dim)
        memory_output = self.read_norm(memory_output + query)
        
        if update_memory:
            # Write to memory
            write_query = query.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]
            updated_slots, _ = self.write_attention(memory, write_query, write_query)
            
            # Gate the update
            gate = self.update_gate(torch.cat([memory, updated_slots], dim=-1))
            updated_memory = gate * updated_slots + (1 - gate) * memory
            updated_memory = self.write_norm(updated_memory)
        else:
            updated_memory = memory
        
        return memory_output, updated_memory


class TemporalContextEncoder(nn.Module):
    """
    Temporal context encoder using recurrent connections
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode temporal context
        
        Args:
            x: [batch_size, seq_len, input_dim] input sequence
            hidden: Optional hidden state from previous step
            
        Returns:
            output: [batch_size, seq_len, input_dim] context-encoded output
            hidden: Updated hidden state
        """
        # Apply LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Project back to input dimension
        output = self.output_proj(lstm_out)
        
        # Residual connection and normalization
        output = self.norm(output + x)
        
        return output, hidden


class AdaptiveContextWindow(nn.Module):
    """
    Adaptive context window that learns to focus on relevant parts of history
    """
    
    def __init__(self, embed_dim: int, max_context_len: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_context_len = max_context_len
        
        # Attention weights for context selection
        self.context_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        
        # Learned queries for different types of context
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.local_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Create adaptive context representation
        
        Args:
            current: [batch_size, embed_dim] current representation
            history: [batch_size, seq_len, embed_dim] historical representations
            
        Returns:
            context_output: [batch_size, embed_dim] context-aware representation
        """
        batch_size = current.shape[0]
        seq_len = history.shape[1]
        
        # Limit context length
        if seq_len > self.max_context_len:
            history = history[:, -self.max_context_len:]
            seq_len = self.max_context_len
        
        # Global context using learned query
        global_query = self.global_query.repeat(batch_size, 1, 1)
        global_context, _ = self.context_attention(global_query, history, history)
        global_context = global_context.squeeze(1)  # [batch_size, embed_dim]
        
        # Local context using current as query
        current_query = current.unsqueeze(1)  # [batch_size, 1, embed_dim]
        local_context, _ = self.context_attention(current_query, history, history)
        local_context = local_context.squeeze(1)  # [batch_size, embed_dim]
        
        # Combine contexts
        combined = torch.cat([current, global_context, local_context], dim=-1)
        gate_weights = self.gate(combined)
        
        # Weighted combination
        context_output = gate_weights * global_context + (1 - gate_weights) * local_context
        context_output = context_output + current  # Residual connection
        
        return context_output


class ContextAwareTokenizer(nn.Module):
    """
    Main context-aware tokenizer with delta encoding and memory
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 context_length: int = 8,
                 use_delta_encoding: bool = True,
                 use_memory: bool = True,
                 memory_size: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.use_delta_encoding = use_delta_encoding
        self.use_memory = use_memory
        
        # Delta encoder
        if use_delta_encoding:
            self.delta_encoder = DeltaEncoder(input_dim, hidden_dim, input_dim)
        
        # Memory module
        if use_memory:
            self.memory = ContextMemory(memory_size, input_dim)
        
        # Temporal context encoder
        self.temporal_encoder = TemporalContextEncoder(input_dim, hidden_dim)
        
        # Adaptive context window
        self.adaptive_context = AdaptiveContextWindow(input_dim, context_length)
    
    def forward(self, 
                inputs: torch.Tensor, 
                memory_state: Optional[torch.Tensor] = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply context-aware tokenization
        
        Args:
            inputs: [batch_size, seq_len, input_dim] input sequence
            memory_state: Optional memory state from previous step
            hidden_state: Optional hidden state from previous step
            
        Returns:
            output: [batch_size, seq_len, input_dim] context-aware output
            updated_memory: Updated memory state
            updated_hidden: Updated hidden state
        """
        batch_size, seq_len, input_dim = inputs.shape
        
        # Apply delta encoding if enabled
        if self.use_delta_encoding:
            delta_encoded = []
            for t in range(seq_len):
                current = inputs[:, t]
                if t == 0:
                    # Use zeros for first timestep
                    previous = torch.zeros_like(current)
                else:
                    previous = inputs[:, t-1]
                
                delta_encoded.append(self.delta_encoder(current, previous))
            
            inputs = torch.stack(delta_encoded, dim=1)
        
        # Apply temporal encoding
        temporal_output, updated_hidden = self.temporal_encoder(inputs, hidden_state)
        
        # Apply memory if enabled
        updated_memory = memory_state
        if self.use_memory:
            memory_output, updated_memory = self.memory(temporal_output, update_memory=True)
            temporal_output = memory_output
        
        # Apply adaptive context for each timestep
        context_outputs = []
        for t in range(seq_len):
            current = temporal_output[:, t]
            if t == 0:
                # No history for first timestep
                context_output = current
            else:
                history = temporal_output[:, :t]
                context_output = self.adaptive_context(current, history)
            
            context_outputs.append(context_output)
        
        output = torch.stack(context_outputs, dim=1)
        
        return output, updated_memory, updated_hidden
