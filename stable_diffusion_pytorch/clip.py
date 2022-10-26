import torch
from torch import nn
from torch.nn import functional as F


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_value = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_value
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x, causal_attention_mask):
        residue = x
        x = self.layernorm_1(x)
        x, _ = self.attention(x, x, x, attn_mask=causal_attention_mask)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

        causal_attention_mask = torch.ones(77, 77, dtype=torch.bool)
        causal_attention_mask.triu_(1)
        self.register_buffer('causal_attention_mask', causal_attention_mask)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state, self.causal_attention_mask)
        output = self.layernorm(state)
        return output