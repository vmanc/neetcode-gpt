from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        tokenized_nums = []
        max_token_len = max(len(t) for t in vocab)
        for num in numbers:
            s = str(num)
            tokens = []
            left = 0
            while (left < len(s)):
                right = min(left + max_token_len, len(s))
                while(right > left and s[left:right] not in vocab):
                    right -= 1
                tokens.append(s[left:right] if right > left else s[left])
                left = max(right, left + 1)
            tokenized_nums.append(tokens)
        return tokenized_nums





    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        max_token_len = max(map(len, vocab), default=1)
        token_count = 0
        left = 0
        while (left < len(text)):
            right = min(left + max_token_len, len(text))
            while(right > left and text[left:right] not in vocab):
                right -= 1
            left = right if right > left else left + 1
            token_count += 1 
        return token_count


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        token_count = self.count_tokens(text, vocab)
        word_count = len(text.split())

        return round(token_count / word_count, 4)
        
