"""
Custom BPE Tokenizer: Trains a 768-vocab BPE on the provided corpus.
Lossless with UTF-8 byte fallback. Massively compresses Hindi/Devanagari text.
"""
import json
import os
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=768):
        self.vocab_size = vocab_size
        self.merges = {} 
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        # Chunking (4000 chars) prevents O(N^2) slowdowns on massive files during eval
        chunk_size = 4000
        result = []
        for c in range(0, len(text), chunk_size):
            chunk = text[c:c+chunk_size]
            tokens = list(chunk.encode("utf-8"))
            while len(tokens) >= 2:
                stats = self.get_stats(tokens)
                best_pair = None
                best_idx = float("inf")
                
                # Apply the earliest learned merge (lowest index)
                for pair in stats:
                    if pair in self.merges and self.merges[pair] < best_idx:
                        best_idx = self.merges[pair]
                        best_pair = pair
                
                if best_pair is None:
                    break
                    
                tokens = self.merge(tokens, best_pair, best_idx)
            result.extend(tokens)
        return result

    def decode(self, ids):
        b = b"".join(self.vocab.get(i, b"") for i in ids)
        return b.decode("utf-8", errors="replace")

    def train(self, text):
        # 180k char sample is enough to quickly learn robust BPE in < 5 seconds
        sample_str = text[:60000] + text[len(text)//2:len(text)//2+60000] + text[-60000:]
        tokens = list(sample_str.encode("utf-8"))
        
        for i in range(256, self.vocab_size):
            stats = self.get_stats(tokens)
            if not stats: 
                break
            best = max(stats, key=stats.get)
            self.merges[best] = i
            self.vocab[i] = self.vocab[best[0]] + self.vocab[best[1]]
            tokens = self.merge(tokens, best, i)

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_merges(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]


def load(path=None):
    tok = BPETokenizer(vocab_size=768)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "bpe_merges.json")
    
    if os.path.exists(save_path):
        tok.load_merges(save_path)
    else:
        corpus_path = os.path.join(base_dir, "../data/train_corpus.txt")
        if os.path.exists(corpus_path):
            text = open(corpus_path, "r", encoding="utf-8").read()
            tok.train(text)
            tok.save(save_path)
    return tok