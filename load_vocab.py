import pickle

with open("tinystories_vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)

with open("tinystories_merges.pkl", "rb") as f:
    loaded_merges = pickle.load(f)

print(loaded_vocab)
print(loaded_merges)