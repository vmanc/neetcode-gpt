import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        tokenized_dataset = raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        batch_indices = torch.randint(0, len(tokenized_dataset) - context_length, (batch_size,)) 
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        X = [tokenized_dataset[i : i + context_length] for i in batch_indices]
        Y = [tokenized_dataset[i + 1 : i + 1 + context_length] for i in batch_indices]
        return X, Y
