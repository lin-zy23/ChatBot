import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from typing import List

import torch
import torch.nn as nn

from data import TextProcessor
from model import ChatbotModel


def load_model(model_path: str,
               tokenizer_path: str,
               device) -> tuple[nn.Module, TextProcessor]:
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    proc = TextProcessor(word_to_idx, idx_to_word)
    model = ChatbotModel(nvoc=len(word_to_idx)).to(device)
    
    model = nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.module
    model.eval()

    return model, proc

def generate_response(model: ChatbotModel, 
                      proc: TextProcessor,
                      history: List[str],
                      device,
                      max_len: int = 512,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      repetition_penalty: float = 1.1) -> str:
    model.eval()
    with torch.no_grad():
        seq = [proc.word_to_idx["<BOS>"]]
        for h in history:
            seq.extend(proc.encode(h))
        inp = torch.tensor([seq], device=device)
        
        puncts = [proc.word_to_idx[p] for p in ['，', '。', '？', '！', '<EOS>', '<UNK>']]
        
        def apply_repetition_penalty(logits: torch.Tensor, generated: List[int]):
            for tok in set(generated):
                logits[tok] /= repetition_penalty
            return logits

        outs = []
        for _ in range(max_len):
            inp = inp[:, -max_len:]
            mask = model._causal_mask(inp.size(1), device)
            
            logits = model(inp, mask)[0, -1] / temperature
            logits = apply_repetition_penalty(logits, outs)
            probs = torch.softmax(logits, dim=-1)
            
            if not outs:
                probs[proc.word_to_idx["<EOS>"]] = 0
                probs = probs / probs.sum()
            
            if outs and outs[-1] in puncts:
                for p in puncts:
                    probs[p] = 0
                probs = probs / probs.sum()
            
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff = cum_probs > top_p
            cutoff[0] = False  # 保证至少有一个 token
            probs[sorted_idx[cutoff]] = 0
            probs = probs / probs.sum()
            
            nxt = torch.multinomial(probs, num_samples=1).item()
            outs.append(nxt)
            
            if nxt == proc.word_to_idx["<EOS>"]:
                break
            
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
        
        return proc.decode(outs)
