from typing import List, Dict

import torch
from torch.utils.data import Dataset


class TextProcessor():
    def __init__(self, word_to_idx: dict = None, idx_to_word: dict = None):
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # 特殊标记
        for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
            self.word_to_idx[tok] = len(self.word_to_idx)
            self.idx_to_word[self.word_to_idx[tok]] = tok
        
        if word_to_idx is not None and idx_to_word is not None:
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
    
    def build_vocab(self, dialogs: List[List[str]], min_freq: int = 2):
        freq: Dict[str, int] = {}
        for q, r in dialogs:
            for sent in (q, r):
                chars = sent.replace(" ", "")
                for ch in chars:
                    freq[ch] = freq.get(ch, 0) + 1

        for ch, cnt in freq.items():
            if cnt >= min_freq and ch not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[ch] = idx
                self.idx_to_word[idx] = ch
    
    def encode(self, sent: str) -> List[int]:
        cleaned = sent.replace(" ", "")
        ids = []
        for ch in cleaned:
            ids.append(self.word_to_idx.get(ch, self.word_to_idx["<UNK>"]))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in {self.word_to_idx["<PAD>"],
                     self.word_to_idx["<BOS>"],
                     self.word_to_idx["<EOS>"],
                     self.word_to_idx["<UNK>"]}:
                continue
            out.append(self.idx_to_word.get(i, ""))
        return "".join(out)


class ChatDataset(Dataset):
    def __init__(self,
                 dialogs: List[List[str]],
                 proc: TextProcessor,
                 max_len: int = 512):
        self.examples: List[torch.Tensor] = []
        self.proc = proc
        self.max_len = max_len
        
        for q, r in dialogs:
            # 拼接：<BOS> + 问句 + 答案 + <EOS>
            seq = [proc.word_to_idx["<BOS>"]] \
                + proc.encode(q) \
                + proc.encode(r) \
                + [proc.word_to_idx["<EOS>"]]
            # 超长截断
            if len(seq) > max_len:
                seq = seq[-max_len:]
            self.examples.append(torch.tensor(seq, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]
