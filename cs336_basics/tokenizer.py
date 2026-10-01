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
    pair_to_pretokens = {}
    pretoken_to_key = {}
    for key, value in counts.items():
        pretoken_str = b"".join(key).decode("utf-8")
        if pretoken_str not in pretoken_to_key:
            pretoken_to_key[pretoken_str] = key
        for l, r in zip(key, key[1:]):
            byte_pair_map[(l, r)] = byte_pair_map[(l, r)] + value if (l, r) in byte_pair_map else value
            if (l, r) in pair_to_pretokens:
                pair_to_pretokens[(l, r)].add(pretoken_str)
            else:
                pair_to_pretokens[(l, r)] = {pretoken_str}
    # print(pretoken_to_key)
    # print(pair_to_pretokens)
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
        # We don't have to traverse through the whole 'counts' dict
        # Instead, traverse only the affected pretokens indicated by pair_to_pretokens
        affected_pretokens = pair_to_pretokens[best_pair].copy()
        for str in affected_pretokens:
            key_in_counts = pretoken_to_key[str]
            appear_counts = counts[key_in_counts]
            new_key = []
            i = 0
            while i < len(key_in_counts):
                pair = key_in_counts[i:i + 2]
                if pair == best_pair:
                    new_pair = pair[0] + pair[1]
                    new_key.append(new_pair)
                    i += 1
                else:
                    new_key.append(pair[0])
                i += 1
            new_key_tuple = tuple(new_key)
            pretoken_to_key[str] = new_key_tuple
            # Update byte_pair_map and pair_to_pretokens 
            # by comparing pair counts before and after merge in this specific affected pretoken
            old_pair_count = {}
            new_pair_count = {}
            for l, r in zip(key_in_counts, key_in_counts[1:]):
                old_pair_count[(l, r)] = old_pair_count[(l, r)] + 1 if (l, r) in old_pair_count else 1
            for l, r in zip(new_key_tuple, new_key_tuple[1:]):
                new_pair_count[(l, r)] = new_pair_count[(l, r)] + 1 if (l, r) in new_pair_count else 1
            all_pairs = old_pair_count.keys() | new_pair_count.keys()
            for pair in all_pairs:
                if new_pair_count.get(pair, 0) == 0:
                    pair_to_pretokens[pair].discard(str)
                if pair in new_pair_count and pair not in old_pair_count:
                    if pair in pair_to_pretokens:
                        pair_to_pretokens[pair].add(str)
                    else:
                        pair_to_pretokens[pair] = {str}
                change = new_pair_count.get(pair, 0) - old_pair_count.get(pair, 0)
                byte_pair_map[pair] = byte_pair_map[pair] + change * appear_counts if pair in byte_pair_map else change * appear_counts
            
            # Update 'counts'
            del counts[key_in_counts]
            counts[new_key_tuple] = appear_counts
        # Update byte_pair_map, deleting all zero-value entries
        byte_pair_map = {
            pair: count
            for pair, count in byte_pair_map.items()
            if count != 0
        }
        # Update pair_to_pretokens, deleting all empty sets
        pair_to_pretokens = {
            pair: pretokens
            for pair, pretokens in pair_to_pretokens.items()
            if pretokens
        }
    return vocab, merges
            


# if __name__ == "__main__":
#     vocab, merges = train_bpe("/home/fox/assignment1-basics/notes.txt", 263, ["<|endoftext|>"])
#     print(vocab)
#     print(merges)