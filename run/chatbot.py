import argparse
import random

import torch

from utils import load_model, generate_response


def interactive_chat(model, proc, device):
    print("\n聊天开始（输入 “exit” 或空行回车结束）")
    chat_history = []
    
    while True:
        q = input("User: ").strip()
        if not q or q.lower() == "exit":
            print("\n结束对话。")
            break
        
        if q[-1] not in [',', '。', '！', '？']:
            q += random.choice(['。', '！', '？', ''])
        
        past = []
        for prev_q, prev_r in chat_history:
            past.extend([prev_q, prev_r])
        past.append(q)
        
        resp = generate_response(model, proc, past, device)
        chat_history.append((q, resp))
        print(f"Chatbot: {resp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatbot_epoch.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model, args.tokenizer, device)
    interactive_chat(model, processor, device)