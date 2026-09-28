from cs336_basics import pretokenization as prt

def train_bpe (
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize vocabulary
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for index, sp_token in enumerate(special_tokens):
        vocab[256 + index] = sp_token.encode("utf-8")
        
    # Pretokenize
    counts = prt.pretokenization(input_path, special_tokens)
    
    # Create byte-pair mapping and pair-pretoken mapping
    byte_pair_map = {}
    
    for key, value in counts.items():
        for l, r in zip(key, key[1:]):
            byte_pair_map[(l, r)] = byte_pair_map[(l, r)] + value if (l, r) in byte_pair_map else value
    
    # Iteratively merge
    
    merges = []
    
    while len(vocab) < vocab_size:
        # Find the most frequent pair in byte_pair_map
        best_pair = max(byte_pair_map, key=lambda pair: (byte_pair_map.get(pair), pair))
        # Record merging for test
        merges.append(best_pair)
        # Add the merged pair to vocab
        new_id = len(vocab)
        vocab[new_id] = best_pair[0] + best_pair[1]
        # Create new 'counts' dict, with the selected pair merged
        merged_counts = {}
        for key, value in counts.items():
            new_key = []
            i = 0
            while i < len(key):
                pair = key[i:i + 2]
                if pair == best_pair:
                    new_key.append(pair[0] + pair[1])
                    i += 1
                else:
                    new_key.append(pair[0])
                i += 1
            new_key_tuple = tuple(new_key)
            merged_counts[new_key_tuple] = value
        counts = merged_counts
        # Create new byte-pair mapping on the new merged count dict
        byte_pair_map = {}
        for key, value in counts.items():
            for l, r in zip(key, key[1:]):
                byte_pair_map[(l, r)] = byte_pair_map[(l, r)] + value if (l, r) in byte_pair_map else value
        # print(counts)
        # print(byte_pair_map)
        # print(merges)
        # loop += 1
    return vocab, merges
            


if __name__ == "__main__":
    vocab, merges = train_bpe("/home/fox/assignment1-basics/notes.txt", 263, ["<|endoftext|>"])
    # print(vocab)
    # print(merges)