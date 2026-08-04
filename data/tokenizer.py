from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        character_list = list(corpus)
        merged_pairs = []

        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        for _ in range(num_merges):
            adjacent_pair_counts = {}
            for i in range(len(character_list) - 1):
                pair = (character_list[i], character_list[i+1])
                adjacent_pair_counts[pair] = adjacent_pair_counts.get(pair, 0) + 1
            
            if not adjacent_pair_counts:
                break
            
            best_pair = min(adjacent_pair_counts, key=lambda p: (-adjacent_pair_counts[p], p))

            index = 0
            merged = []
            while (index < len(character_list)):
                if (
                    index < len(character_list) - 1 
                    and character_list[index] == best_pair[0]
                    and character_list[index+1] == best_pair[1]
                ):
                    merged.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged.append(character_list[index])
                    index += 1
            
            character_list = merged
            merged_pairs.append([best_pair[0], best_pair[1]])
        
        return merged_pairs 
        
