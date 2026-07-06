import flax.linen as nn
import jax.numpy as jnp

class TransformerContextEncoder(nn.Module):
    """Processes a sequence of state-action embeddings to infer a goal."""
    output_size: int
    num_heads: int = 4
    num_layers: int = 2
    d_model: int = 256
    max_len: int = 2048

    @nn.compact
    def __call__(self, sa_repr_sequence):
        # sa_repr_sequence has shape (batch_size, seq_len, repr_dim)
        
        # Project input to d_model
        x = nn.Dense(self.d_model, name="input_projection")(sa_repr_sequence)

        # Add learnable positional embeddings
        seq_len = x.shape[1]
        pos_embedding = self.param('pos_embedding', nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'), (self.max_len, self.d_model))
        x = x + pos_embedding[None, :seq_len, :]  # Add broadcasting dim for batch and slice to seq_len

        # Transformer Encoder Layers
        for i in range(self.num_layers):
            # Multi-head self-attention
            x_norm = nn.LayerNorm(name=f"ln1_layer_{i}")(x)
            attn_output = nn.SelfAttention(num_heads=self.num_heads, name=f"attention_layer_{i}")(x_norm)
            x = x + attn_output
            
            # Feed-forward
            x_norm = nn.LayerNorm(name=f"ln2_layer_{i}")(x)
            ff_output = nn.Dense(self.d_model * 4, name=f"ff1_layer_{i}")(x_norm)
            ff_output = nn.gelu(ff_output)
            ff_output = nn.Dense(self.d_model, name=f"ff2_layer_{i}")(ff_output)
            x = x + ff_output

        # Aggregate the sequence. Mean pooling is a simple and effective choice.
        trajectory_embedding = jnp.mean(x, axis=1)

        # Output head to predict goal distribution parameters
        output = nn.Dense(self.output_size, name="output_head")(trajectory_embedding)
        return output 