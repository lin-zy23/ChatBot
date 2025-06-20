import os
import glob
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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence

from data import TextProcessor, ChatDataset
from model import ChatbotModel


def init_distributed():
    dist.init_process_group(backend="nccl", init_method="env://",)
    local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def collate_fn(batch: List[torch.Tensor], pad_idx: int) -> torch.Tensor:
    seqs = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return seqs  # [batch, L_max]

def evaluate_perplexity(model: ChatbotModel, dataloader, 
                        criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loop = tqdm(dataloader, desc="Step", total=len(dataloader), 
                    leave=False, mininterval=60, miniters=100)
    
    with torch.no_grad():
        for seqs in loop:
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
                criterion,
                device,
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
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (train_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    for epoch in trange(1, num_epochs + 1, desc="Epoch"):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc="Step", total=len(train_loader), 
                    leave=False, mininterval=60, miniters=100)
        
        for step, seqs in enumerate(loop, 1):
            seqs = seqs.to(device)           # [B, L]
            inputs  = seqs[:, :-1]           # [B, L-1]
            targets = seqs[:, 1:]            # [B, L-1]
            
            if isinstance(model, nn.parallel.DistributedDataParallel):
                mask = model.module._causal_mask(inputs.size(1), device)
            else:
                mask = model._causal_mask(inputs.size(1), device)
    
            optimizer.zero_grad()
            with autocast():
                logits = model(inputs, mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1))
           
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            if step % print_every == 0:
                avg = total_loss / print_every
                if (not distributed) or (dist.get_rank() == 0):
                    print(f"\n[Epoch {epoch}] Step {step}/{len(train_loader)}  AvgLoss={avg:.4f}")
                total_loss = 0.0
                torch.save(model.state_dict(), f"chatbot_epoch{epoch}.pt")

        if not distributed or dist.get_rank() == 0:
            torch.save(model.state_dict(), f"chatbot_epoch{epoch}.pt")
            ppl = evaluate_perplexity(model, valid_loader, criterion, device)
            print(f"\n→ Epoch {epoch} completed. Validation Perplexity: {ppl:.2f}\n")
        
        if distributed:
            dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    
    if args.distributed:
        local_rank = init_distributed()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    all_files = sorted(glob.glob("data/chinese-cosmopedia/data/*.parquet"))
    train_paths = all_files[:-1]
    valid_paths = all_files[-1:]
    
    with open("tokenizer.json", "r", encoding="utf-8") as fr:
        word_to_idx = json.load(fr)
    
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    proc = TextProcessor(word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    global pad_idx
    pad_idx = proc.word_to_idx["<PAD>"]
    
    train_dataset = ChatDataset(train_paths, proc, max_len=512)
    valid_dataset = ChatDataset(valid_paths, proc, max_len=512)
    
    if args.distributed:
        train_loader = DataLoader(train_dataset,
                                  batch_size=32,
                                  num_workers=32,
                                  pin_memory=True,
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
        
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=32,
                                  num_workers=32,
                                  pin_memory=True,
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
    
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=32,
                                  num_workers=32,
                                  pin_memory=True,
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
        
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=32,
                                  num_workers=32,
                                  pin_memory=True,
                                  collate_fn=lambda b: collate_fn(b, pad_idx))
    
    model = ChatbotModel(nvoc=len(proc.word_to_idx),
                         dim=768, nhead=12, num_layers=12,
                         max_len=512, dropout=0.1).to(device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print("▶ Start training\n")
    train_model(model, train_loader, valid_loader, criterion, device, 
                num_epochs=args.epochs, print_every=100000, 
                distributed=args.distributed)
    print("✅ Training completed.\n")
    
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
