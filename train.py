import os
import json
import math
import argparse
from tqdm import trange, tqdm
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from data import TextProcessor, ChatDataset
from model import ChatbotModel


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def collate_fn(batch: List[torch.Tensor], pad_idx: int) -> torch.Tensor:
    seqs = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return seqs  # [batch, L_max]

def load_pairs_from_jsonl(path: str) -> List[List[str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            arr = json.loads(line.strip())
            for i in range(len(arr) - 1):
                q, r = arr[i], arr[i + 1]
                pairs.append([q, r])
    return pairs

def evaluate_perplexity(model: ChatbotModel, dataloader, 
                        criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for seqs in dataloader:
            seqs = seqs.to(device)
            inputs  = seqs[:, :-1]
            targets = seqs[:, 1:]
            
            if isinstance(model, nn.parallel.DistributedDataParallel):
                mask = model.module._causal_mask(inputs.size(1), device)
            else:
                mask = model._causal_mask(inputs.size(1), device)
                
            logits = model(inputs, mask)  # [B, L-1, V]
            
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))
            ntokens = (targets != pad_idx).sum().item()
            total_loss += loss.item() * ntokens
            total_tokens += ntokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def train_model(model: ChatbotModel,
                train_loader,
                valid_loader,
                criterion: nn.Module,
                device: torch.device,
                num_epochs: int = 10,
                print_every: int = 1000,
                distributed: bool = False):    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)
    
    train_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * train_steps)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(train_steps - current_step) / float(max(1, train_steps - warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    for epoch in trange(1, num_epochs + 1, desc="Epoch"):
        model.train()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        loop = tqdm(train_loader, desc="Step", 
                    total=len(train_loader), leave=False)
        
        for step, seqs in enumerate(loop, 1):
            seqs = seqs.to(device)           # [B, L]
            inputs  = seqs[:, :-1]           # [B, L-1]
            targets = seqs[:, 1:]            # [B, L-1]
            
            if isinstance(model, nn.parallel.DistributedDataParallel):
                mask = model.module._causal_mask(inputs.size(1), device)
            else:
                mask = model._causal_mask(inputs.size(1), device)
    
            optimizer.zero_grad()
            logits = model(inputs, mask)     # [B, L-1, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if step % print_every == 0:
                avg = total_loss / print_every
                if (not distributed) or (dist.get_rank() == 0):
                    print(f"\n[Epoch {epoch}] Step {step}/{len(train_loader)}  AvgLoss={avg:.4f}")
                total_loss = 0.0

        if not distributed or dist.get_rank() == 0:
            ppl = evaluate_perplexity(model, valid_loader, criterion, device)
            print(f"\n→ Epoch {epoch} completed. Validation Perplexity: {ppl:.2f}\n")
            
            torch.save(model.state_dict(), f"chatbot_epoch{epoch}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    if args.distributed:
        local_rank = init_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device_ids = list(map(int, args.devices.split(',')))
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    train_pairs = load_pairs_from_jsonl(
        "data/lccc/lccc_base_train/LCCC-base_train.jsonl")
    valid_pairs = load_pairs_from_jsonl(
        "data/lccc/lccc_base_valid/LCCC-base_valid.jsonl")
    
    # 构建词汇表
    proc = TextProcessor()
    proc.build_vocab(train_pairs, min_freq=3)
    global pad_idx
    pad_idx = proc.word_to_idx["<PAD>"]
    
    with open("tokenizer.json", "w", encoding="utf-8") as fw:
        json.dump(proc.word_to_idx, fw, ensure_ascii=False, indent=2)
    
    train_dataset = ChatDataset(train_pairs, proc, max_len=512)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=32, 
                                  sampler=train_sampler, 
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
    else:
        train_loader = DataLoader(train_dataset, 
                                  batch_size=32, 
                                  shuffle=True, 
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
    
    valid_dataset = ChatDataset(valid_pairs, proc, max_len=512)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=32,
                              collate_fn=lambda b: collate_fn(b, pad_idx))
    
    model = ChatbotModel(nvoc=len(proc.word_to_idx),
                         dim=512, nhead=8, num_layers=6,
                         max_len=512, dropout=0.1).to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print("▶ Start training")
    train_model(model, train_loader, valid_loader, criterion, device, 
                num_epochs=args.epochs, print_every=10000, 
                distributed=args.distributed)
    
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()