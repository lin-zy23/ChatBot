import os
import math
from typing import List, Iterator

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class TextProcessor:
    def __init__(self, word_to_idx: dict = None, idx_to_word: dict = None):
        self.word_to_idx = {}
        self.idx_to_word = {}
        for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
            self.word_to_idx[tok] = len(self.word_to_idx)
            self.idx_to_word[self.word_to_idx[tok]] = tok

        if word_to_idx is not None:
            self.word_to_idx = word_to_idx
            if idx_to_word is not None:
                self.idx_to_word = idx_to_word

    def encode(self, sent: str) -> List[int]:
        ids = []
        for ch in sent:
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


def _count_file_chunks(args):
    path, word_to_idx, max_len, batch_rows = args
    proc = TextProcessor(word_to_idx=word_to_idx, idx_to_word=None)
    pf = pq.ParquetFile(path)

    file_count = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
        for text in batch.to_pydict()["text"]:
            clean = text.replace("#", "").replace("*", "")
            seq_len = len(proc.encode(clean)) + 2
            file_count += math.ceil(seq_len / max_len)
    
    return file_count


class ChatDataset(IterableDataset):
    def __init__(self,
                 parquet_paths: List[str],
                 proc: TextProcessor,
                 max_len: int = 512,
                 batch_rows: int = 1024):
        super().__init__()
        self.parquet_paths = list(parquet_paths)
        self.proc = proc
        self.max_len = max_len
        self.batch_rows = batch_rows
        
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        total_files = len(self.parquet_paths)
        if self.world_size > 1:
            per_rank = int(math.ceil(total_files / float(self.world_size)))
            start_file = self.rank * per_rank
            end_file = min(start_file + per_rank, total_files)
            self.rank_paths = self.parquet_paths[start_file:end_file]
        else:
            self.rank_paths = self.parquet_paths
        
        tasks = [(path, proc.word_to_idx, max_len, batch_rows)
                 for path in self.rank_paths]

        total_chunks = 0
        with ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as exe:
            futures = [exe.submit(_count_file_chunks, task) for task in tasks]
            for f in tqdm(as_completed(futures),
                          total=len(futures),
                          desc=f"Rank {self.rank} counting chunks",
                          unit="file",
                          leave=False):
                total_chunks += f.result()
        self._length = total_chunks
    
    def __len__(self) -> int:
        return self._length
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        if worker_info is None:
            paths = self.rank_paths
        else:
            per_worker = int(math.ceil(len(self.rank_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.rank_paths))
            paths = self.rank_paths[start:end]
        
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=self.batch_rows, columns=["text"]):
                for text in batch.to_pydict()["text"]:
                    clean = text.replace("#", "").replace("*", "")
                    ids = self.proc.encode(clean)
                    seq = [self.proc.word_to_idx["<BOS>"]] + ids + [self.proc.word_to_idx["<EOS>"]]
                    
                    for i in range(0, len(seq), self.max_len):
                        chunk = seq[i: i + self.max_len]
                        yield torch.tensor(chunk, dtype=torch.long)
