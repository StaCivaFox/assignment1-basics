## Pretokenization
1.Take in a file, open and read from it.2.Split on <|endoftext|> using re.split.3.Use re.finditer to find and iterate through pre-tokens found by regex pattern,count appearance and store in a dict[tuple[bytes, ...], int]. 
## train_bpe
1.Pre-tokenize the corpus,
  receiving a counts  dict from tuple of bytes to int count.2.iterate
  through the counts dict once, creating a dict mapping from
  tuple(bytes, bytes) to int count.perhaps we can choose some data
  structures like heap to keep the most frequent pair at top.3.find the
  most frequent bytes pair, and then iterate through the pre-token
  dict for another pass.whenever we find that pair, merge
  them.specifically, replace those two bytes object with a new one in
  that tuple key in the pre-token dict.4.iterate through pre-token
  dict for yet another pass and create a new dict[tuple[bytes, bytes],
  int],and repeat step 2-4

  p | a | b | p len = 4
  p | a | b len = 3
  p | ab | p | a | q | r

  # generate
  while not reach max token or not reach eos:
    1.make one inference pass through model, producing logits shaped (batch_size, seq_len, vocab_size)
    2.pass logits through a temperature-controlled softmax, producing probabilities shaped (batch_size, seq_len, vocab_size)
    3.get the predicted probabilities distribution of the (seq_len - 1)th token of each batch, getting q shaped (batch_size, vocab_size) where vocab_size describes a next-token distribution for each batch
    4. use top-p to truncate and renormalize the last axis of q(batch_size, vocab_size), getting a probability distribution for sampling
    5.sample using torch.multinomial, getting (batch_size, 1), with one next token id for each batch
    6.concatenate the newly predicted token id to x and slice the last seq_len entries, prepare for next pass