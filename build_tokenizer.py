import glob
import json

import pyarrow.parquet as pq
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from data import TextProcessor


def count_file_chars(path: str, batch_rows: int = 512) -> Counter:
    freq = Counter()
    pf = pq.ParquetFile(path)
    
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
        arr = batch.to_pydict()["text"]
        for txt in arr:
            clean = txt.replace("#", "").replace("*", "")
            for ch in clean:
                freq[ch] += 1
    
    return freq


def main():
    all_files = sorted(glob.glob("data/chinese-cosmopedia/data/*.parquet"))
    train_files = all_files[:-1]
    
    total_freq = Counter()
    with ProcessPoolExecutor(max_workers=32) as exe:
        futures = {exe.submit(count_file_chars, path): path for path in train_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Counting chars"):
            total_freq.update(fut.result())
    
    min_freq = 3
    
    proc = TextProcessor()
    for ch, cnt in tqdm(total_freq.items(), total=len(total_freq), desc="Building vocab"):
        if cnt >= min_freq and ch not in proc.word_to_idx:
            idx = len(proc.word_to_idx)
            proc.word_to_idx[ch] = idx
            proc.idx_to_word[idx] = ch
    
    with open("tokenizer.json", "w", encoding="utf-8") as fw:
        json.dump(proc.word_to_idx, fw, ensure_ascii=False, indent=2)

    print(f"✅ tokenizer.json generated, vocab size = {len(proc.word_to_idx)}")


if __name__ == "__main__":
    main()
