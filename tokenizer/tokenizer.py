class Tokenizer:
    def __init__(self, tokens, vocab_size) -> None:
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256 # desired vocab size vs current vocab size
        self.tokens = tokens
        self.ids = []
        self.vocab = self._build_vocab()
    
    def get_ids(self):
        return self.ids
    
    def get_vocab(self):
        return self.vocab
    
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        # in the list of ints (ids), replace all consecutive occurences of pair with the new token id
        newTokenIds = []
        i = 0
        while i < len(ids):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newTokenIds.append(idx)
                i += 2
            else:
                newTokenIds.append(ids[i])
                i += 1
        return newTokenIds
    
    def _build_vocab(self):
        tokenIds = list(self.tokens)
        merges = {} # (int, int) -> int
        for i in range(self.num_merges):
            stats = self._get_stats(tokenIds)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"Minting new token: {idx}, by mergin merging pair: {pair}")
            tokenIds = self._merge(tokenIds, pair, idx)
            merges[pair] = idx 
        self.ids = tokenIds
        return merges
    
    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.vocab.get(p, float("inf")))
            if pair not in self.vocab:
                break # nothing else can be merged
            idx = self.vocab[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.vocab.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        # given ids (list of integers), return Python string
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    
    def get_compression_stats(self):
        return {
            'orignal_token_count': len(self.tokens),
            'final_token_count': len(self.ids),
            'compression_ratio': len(self.tokens) / len(self.ids),
            'space_savings': (len(self.tokens) - len(self.ids)) * 100 / len(self.tokens)
        }