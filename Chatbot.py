import argparse
import torch
from call import load_model, generate_response


def interactive_chat(model, proc, device):
    print("聊天开始（输入 “exit” 结束）")
    while True:
        q = input("User: ").strip()
        if q == "" or q.lower() == "exit":
            print("结束对话。")
            break
        resp = generate_response(model, proc, q, device)
        print(f"Chatbot: {resp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatbot.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model, args.tokenizer, device)
    interactive_chat(model, processor, device)