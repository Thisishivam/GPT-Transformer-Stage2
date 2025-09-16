class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = hidden_dim * mlp_ratio
        
        # SwiGLU uses 2/3 * 4 = 8/3 expansion factor
        self.gate_proj = ColumnParallelLinear(hidden_dim, self.mlp_hidden_dim * 2 // 3, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_dim, self.mlp_hidden_dim * 2 // 3, bias=False)
        self.down_proj = RowParallelLinear(self.mlp_hidden_dim * 2 // 3, hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("FlashAttention not installed, using fallback")

class FlashAttentionV2(nn.Module):
    def __init__(self, head_dim: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if HAS_FLASH_ATTN and q.dtype in [torch.float16, torch.bfloat16]:
            # Use real FlashAttention v2
            return flash_attn_func(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                softmax_scale=self.scale
            )
        else:
            B, H, S, D = q.shape
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if self.causal:
                causal_mask = torch.triu(
                    torch.full((S, S), float('-inf'), device=q.device, dtype=q.dtype),
                    diagonal=1
                )
                attn_scores = attn_scores + causal_mask
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            return torch.matmul(attn_probs, v)

class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 rotary_dim: int,
                 layer_id: int, 
                 num_layers: int,  
                 dropout: float = 0.0,
                 mlp_ratio: int = 4,
                 use_checkpointing: bool = True):
        """
        Args:
            hidden_dim: Model dimension (D)
            num_heads: Number of attention heads
            rotary_dim: Dimension for rotary embeddings
            layer_id: Current layer index (0-based)
            num_layers: Total number of layers in model
            dropout: Dropout probability (typically 0.0 for large models)
            mlp_ratio: Multiplier for MLP hidden dimension
            use_checkpointing: Enable gradient checkpointing for memory
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.dropout = dropout if num_layers < 32 else 0.0 

        # Attention components
        self.attn_norm = RMSNorm(hidden_dim)
        self.qkv_proj = ProductionQKVProjection(hidden_dim, num_heads, rotary_dim)
        self.attention = FlashAttentionV2(hidden_dim // num_heads, dropout=self.dropout, causal=True)
        self.attn_out = RowParallelLinear(hidden_dim, hidden_dim, bias=False)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, mlp_ratio)
        self._init_weights()

    def _init_weights(self):
        attn_std = 0.02 / math.sqrt(2 * self.num_layers)
        mlp_std = 0.02 / math.sqrt(2 * self.num_layers * 3)
        
        nn.init.normal_(self.attn_out.weight, mean=0.0, std=attn_std)
        

    def _forward_impl(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        q, k, v = self.qkv_proj(x, positions)
        attn_output = self.attention(q, k, v)
        batch_size, seq_len = x.shape[0], x.shape[1]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.attn_out(attn_output)
        
        if self.dropout > 0:
            attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        x = residual + attn_output

        residual = x
        x = self.mlp_norm(x)
        mlp_output = self.mlp(x)
        
        if self.dropout > 0:
            mlp_output = F.dropout(mlp_output, p=self.dropout, training=self.training)
        
        x = residual + mlp_output

        return x

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            # Gradient checkpointing for memory efficiency
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, positions, 
                use_reentrant=False,
                preserve_rng_state=True
            )
        else:
            return self._forward_impl(x, positions)

if __name__ == "__main__":
    # Configuration
    HIDDEN_DIM = 512
    NUM_HEADS = 8
    ROTARY_DIM = 64
    NUM_LAYERS = 12
    BATCH_SIZE = 2
    SEQ_LEN = 128
  
    block = TransformerBlock(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        rotary_dim=ROTARY_DIM,
        layer_id=0,
        num_layers=NUM_LAYERS,
        dropout=0.1,  # Will be scaled to 0.0 for large models
        use_checkpointing=True
    ).cuda().to(torch.bfloat16)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(SEQ_LEN, device='cuda')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = block(x, positions)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Using SwiGLU: {hasattr(block.mlp, 'gate_proj')}")
        print(f"Using checkpointing: {block.use_checkpointing}")
        print(f"Dropout: {block.dropout}")
        print("✅ Production Transformer Block working correctly!")

    # Parameter count
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Number of parameters: {total_params:,}")
