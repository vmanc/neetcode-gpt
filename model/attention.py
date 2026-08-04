import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        attention_scores = torch.matmul(Q, K.mT) / math.sqrt(Q.shape[-1])
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        mask = torch.tril(torch.ones(Q.shape[-2], Q.shape[-2]))
        masked_scores = attention_scores.masked_fill(mask==0, float('-inf'))
        
        # 4. Apply softmax(dim=2) to masked scores
        softmax_scores = nn.functional.softmax(masked_scores, dim=2)

        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(torch.matmul(softmax_scores, V), decimals=4)
        