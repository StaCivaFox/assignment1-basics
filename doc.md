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