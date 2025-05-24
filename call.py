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
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, proc

def generate_response(model: ChatbotModel, 
                      proc: TextProcessor,
                      history: List[str],
                      device,
                      max_len: int = 50,
                      temperature: float = 0.8,
                      top_p: float = 0.9) -> str:
    model.eval()
    with torch.no_grad():
        seq = [proc.word_to_idx["<BOS>"]]
        for h in history:
            seq.extend(proc.encode(h))
        inp = torch.tensor([seq], device=device)
        
        outs = []
        for _ in range(max_len):
            mask = model._causal_mask(inp.size(1), device)
            logits = model(inp, mask)[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_p, dim=0)
            cutoff = cumprobs > top_p
            cutoff[0] = False
            probs[sorted_idx[cutoff]] = 0
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1).item()
            if nxt == proc.word_to_idx["<EOS>"]:
                break
            outs.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
        return proc.decode(outs)