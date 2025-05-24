import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, dropout: float, attn_drop: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=attn_drop)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [seq_len, batch, dim]
        a = self.ln_1(x)
        attn_out, _ = self.attn(a, a, a, attn_mask=mask, need_weights=False)
        x = x + self.drop_attn(attn_out)

        m = self.ln_2(x)
        x = x + self.mlp(m)
        return x


class ChatbotModel(nn.Module):
    def __init__(self,
                 nvoc: int,
                 dim: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 attn_drop: float = 0.1,
                 emb_drop: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        self.token_emb = nn.Embedding(nvoc, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(emb_drop)
        
        self.h = nn.ModuleList([
            TransformerBlock(dim, nhead, dropout, attn_drop)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim, eps=1e-5)
        self.lm_head = nn.Linear(dim, nvoc, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, L: int, device):
        return torch.triu(torch.full((L, L), float("-inf"), device=device),
                          diagonal=1)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # idx: [batch, seq_len]
        B, L = idx.size()
        assert L <= self.max_len, "Sequence length exceeds model capacity."
        
        pos_indices = torch.arange(L, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos_indices)
        x = self.drop(x)
        
        x = x.transpose(0, 1)
        
        if mask is None:
            mask = self._causal_mask(L, idx.device)
        
        for block in self.h:
            x = block(x, mask)
        
        x = x.transpose(0, 1)       # [B, L, D]
        x = self.ln_f(x)            # [B, L, D]
        logits = self.lm_head(x)    # [B, L, nvoc]
        return logits