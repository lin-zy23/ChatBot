import argparse
import random

import torch

from call import load_model, generate_response


def interactive_chat(model, proc, device):
    print("\n聊天开始（输入 “exit” 或空行回车结束）")
    while True:
        q = input("User: ").strip()
        if q == "" or q.lower() == "exit":
            print("\n结束对话。")
            break
        
        if q[-1] not in [',', '。', '！', '？']:
            q += random.choice(['。', '！', '？', ''])
        
        resp = generate_response(model, proc, q, device)
        print(f"Chatbot: {resp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatbot.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model, args.tokenizer, device)
    interactive_chat(model, processor, device)