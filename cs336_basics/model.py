import torch
import torch.nn as nn
from einops import einsum, rearrange
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (out_features + in_features))
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype), 0.0, std, -3 * std, 3 * std), 
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 0.0, 1.0, -3, 3), 
            requires_grad=True
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # batch_size sequence_length -> batch_size sequence_length embedding_dim
        return self.weights[token_ids, :]

class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(end=max_seq_len, device=device).unsqueeze(dim=-1)
        denominators = torch.linspace(start=1, end=d_k // 2, steps=d_k // 2, device=device).unsqueeze(dim=0)
        angles = positions / (theta ** ((2 * denominators - 2) / d_k)) # max_seq_len (d_k // 2)
        sines = torch.sin(angles)
        cosines = torch.cos(angles)
        rotation_block_row1 = torch.stack((cosines, -sines), dim=-1) # max_seq_len (d // 2) 2
        rotation_block_row2 = torch.stack((sines, cosines), dim=-1)
        # A matrix's rows correspond to output coordinates, as to "how many times the computation happens"
        # Its columns correspond to input coordinates, as to "what is computed in each time"
        # The first stacking tensors along dim=-1 is like combining column vectors left to right, forming a row
        # The second stacking tensors along dim=-2 is like combining row vectors top to bottom, forming a column
        rotation = torch.stack((rotation_block_row1, rotation_block_row2), dim=-2) # max_seq_len (d // 2) 2 2
        self.register_buffer("rotation", rotation, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        selected = self.rotation[token_positions] # ... seq_len (d//2) 2 2
        in_dtype = x.dtype
        x = x.to(selected.dtype)
        # Group x by pairs; x_paired is shaped ... seq_len (d//2) 2 
        x_paired = rearrange(x, "... (pair_count coor) -> ... pair_count coor", coor=2)
        roted = einsum(selected, x_paired, "... out_coor in_coor, ... in_coor -> ... out_coor")
        return rearrange(roted, "... pair_count coor -> ... (pair_count coor)").to(in_dtype)


def softmax(x: torch.Tensor, dim: int):
    x = x - x.amax(dim=dim, keepdim=True)
    row_sum = torch.exp(x).sum(dim=dim, keepdim=True)
    return torch.exp(x) / row_sum

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, positional_encoder: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.positional_encoder: RotaryPositionalEmbedding | None = positional_encoder

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        proj_q = self.W_q(x)
        proj_k = self.W_k(x)
        proj_v = self.W_v(x)
        seq_len = x.shape[-2]
        # Expose the head axis
        proj_q = rearrange(proj_q, "batch_size ... (heads d_k) -> batch_size heads ... d_k", heads = self.num_heads)
        proj_k = rearrange(proj_k, "batch_size ... (heads d_k) -> batch_size heads ... d_k", heads = self.num_heads)
        proj_v = rearrange(proj_v, "batch_size ... (heads d_v) -> batch_size heads ... d_v", heads = self.num_heads)
        # Apply RoPE
        if self.positional_encoder is not None:
            # If we have explicit token positions
            if token_positions is not None:
                # Align up head axis for rotation matrix index
                token_positions = token_positions.unsqueeze(dim=-2)
            else:
                token_positions = torch.arange(end=x.shape[-2]) # x.shape[-2] is seq_len
            proj_q = self.positional_encoder(proj_q, token_positions)
            proj_k = self.positional_encoder(proj_k, token_positions)
        # Causal mask
        mask = torch.tril(torch.full((seq_len, seq_len), True, device=x.device))
        # Calculate attention
        attention_output = scaled_dot_product_attention(Q=proj_q, K=proj_k, V=proj_v, mask=mask)
        # Concatenate all heads
        attention_output = rearrange(attention_output, "batch_size heads ... d_k -> batch_size ... (heads d_k)")
        return self.W_o(attention_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_encoder: RotaryPositionalEmbedding | None = None):
        super().__init__()
        # rms norm
        self.norm1 = rmsnorm(d_model=d_model)
        # multi-head attention
        self.mha = CausalMultiheadSelfAttention(d_model=d_model, num_heads=num_heads, positional_encoder=positional_encoder)
        # rms norm
        self.norm2 = rmsnorm(d_model=d_model)
        # FFN
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        res_sublayer1 = x + self.mha(self.norm1(x))
        res_sublayer2 = res_sublayer1 + self.ffn(self.norm2(res_sublayer1))
        return res_sublayer2


class TransformerLM(nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            rope_theta: float | None = 10000.0
        ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoder = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, positional_encoder=self.positional_encoder) for _ in range(num_layers)
        ])
        self.norm = rmsnorm(d_model=d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_embedding(self.norm(x))
        return x


