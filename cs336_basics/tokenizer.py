import pickle
from collections.abc import Iterable, Iterator
from cs336_basics import train_bpe, pretokenization

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merged_pair_to_pos = {}
        for index, pair in enumerate(self.merges):
            self.merged_pair_to_pos[pair] = index
        # Add unseen special tokens to vocab
        vocab_size = len(vocab)
        if special_tokens:
            offset = 0
            for str in special_tokens:
                str_encoded = str.encode("utf-8")
                if str_encoded not in self.vocab.values():
                    self.vocab[vocab_size + offset] = str_encoded
                    offset += 1
        # print(self.vocab)


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # The provided tests don't call this method;
        # Therefore, assume vocab and merges is saved as pickle.dump file in terms of convenience
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges: list[tuple[bytes, bytes]] = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Pre-tokenize
        pretokens_list = pretokenization.pretokenization_for_tokenizer(text, self.special_tokens)
        # print(pretokens_list)
        encoded_text = []
        vocab_to_id = {value: key for key, value in self.vocab.items()}
        pretoken_to_id = {} #dict[tuple[bytes, ...], list[int]], the key being the original not-merged tuple
        # Iterate through each pre-token and encode it
        for pretoken in pretokens_list:
            if pretoken in pretoken_to_id:
                token_id = pretoken_to_id[pretoken]
                encoded_text.extend(token_id)
                continue
            token_id = []
            tmp_pretoken = pretoken
            # Iteratively merge each pretoken's inner bytes
            while True:
                merged_pretoken = []
                # Choose the earliest-appearing pair to merge
                best_pair = None
                for l, r in zip(tmp_pretoken, tmp_pretoken[1:]):
                    # print(l, r)
                    if (l, r) in self.merged_pair_to_pos and \
                    (best_pair is None or \
                     self.merged_pair_to_pos[(l, r)] < self.merged_pair_to_pos[best_pair]):
                        best_pair = (l, r)
                if best_pair is None:
                    break
                i = 0
                # Merge for one round
                while i < len(tmp_pretoken):
                    pair = tmp_pretoken[i:i + 2]
                    if pair == best_pair:
                        new_pair = pair[0] + pair[1]
                        merged_pretoken.append(new_pair)
                        i += 1
                    else:
                        merged_pretoken.append(pair[0])
                    i += 1
                tmp_pretoken = tuple(merged_pretoken)
            # When leaving the loop, tmp_pretoken is a tuple holding bytes indexed into vocab
            for token in tmp_pretoken:
                token_id.append(vocab_to_id[token])
            pretoken_to_id[pretoken] = token_id
            encoded_text.extend(token_id)
        return encoded_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for str in iterable:
            encoding = self.encode(str)
            yield from encoding

    def decode(self, ids: list[int]) -> str:
        byte_parts = map(lambda token_id: self.vocab[token_id], ids)
        result_str = b"".join(byte_parts).decode("utf-8", errors="replace")
        return result_str

                





if __name__ == "__main__":
    def test_handout_encoding():
        vocab = {
            0: b" ",
            1: b"a",
            2: b"c",
            3: b"e",
            4: b"h",
            5: b"t",
            6: b"th",
            7: b" c",
            8: b" a",
            9: b"the",
            10: b" at",
        }

        merges = [
            (b"t", b"h"),
            (b" ", b"c"),
            (b" ", b"a"),
            (b"th", b"e"),
            (b" a", b"t"),
        ]

        tokenizer = Tokenizer(vocab, merges)

        # Check individual pretokens, preserving leading spaces.
        assert tokenizer.encode("the") == [9]
        assert tokenizer.encode(" cat") == [7, 1, 5]
        assert tokenizer.encode(" ate") == [10, 3]

        # Check the complete input.
        assert tokenizer.encode("the cat ate") == [9, 7, 1, 5, 10, 3]
        print(tokenizer.encode("the cat ate"))

    def test_priority_encoding():
            vocab = {
                0: b" ",
                1: b"a",
                2: b"b",
                3: b"c",
                4: b"ab",
                5: b"abc",
            }
    
            merges = [
                (b"a", b"b"),
                (b"ab", b"c"),
            ]
    
            tokenizer = Tokenizer(vocab, merges)
    
    
            print(tokenizer.encode("abc"))

    # test_handout_encoding()
    test_priority_encoding()
    # vocab, merges = train_bpe.train_bpe("/home/fox/assignment1-basics/notes.txt", 263, ["<|endoftext|>"])
    # tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>", "<PAD>"])
    # tokenizer.encode("low low low low low lower lower widest widest widest newest newest newest newest newest newest")

