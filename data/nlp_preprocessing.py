import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined_list = list(positive) + list(negative)
        all_words = []  
        for sentence in combined_list:
            all_words.extend(sentence.split())
        
        vocab = {word: i for i, word in enumerate(sorted(set(all_words)), start=1)}
        encoded = [torch.tensor([vocab[word] for word in sentence.split()]) for sentence in combined_list]
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True).float()