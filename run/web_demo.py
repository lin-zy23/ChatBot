import argparse
import random

import torch
import gradio as gr

from utils import load_model, generate_response


PUNCTS = ['。', '！', '？']

def predict(user_input: str,
            chat_history: list,
            model, proc, device):
    original = user_input
    
    prompt_input = original
    # if prompt_input and prompt_input[-1] not in [',','。','！','？','~']:
    #     prompt_input += random.choice(PUNCTS)
    
    past = [msg["content"] for msg in chat_history] + [prompt_input]
    response = generate_response(model, proc, past, device)
    
    chat_history = chat_history + [
        {"role": "user", "content": original},
        {"role": "assistant", "content": response}
    ]

    return chat_history, chat_history

def regenerate(chat_history: list, model, proc, device):
    if len(chat_history) < 2:
        return [], []
    
    last_assistant = chat_history.pop()
    last_user      = chat_history.pop()
    
    original = last_user["content"]
    
    prompt_input = original
    if prompt_input and prompt_input[-1] not in [',','。','！','？','~']:
        prompt_input += random.choice(PUNCTS)
    
    past = [msg["content"] for msg in chat_history] + [prompt_input]
    new_resp = generate_response(model, proc, past, device)
    
    chat_history += [
        last_user,
        {"role": "assistant", "content": new_resp}
    ]

    return chat_history, chat_history


def clear_history():
    return [], []


def deploy(model_path: str, tokenizer_path: str, device):
    model, proc = load_model(model_path, tokenizer_path, device)
    
    with gr.Blocks(title="Gradio") as demo:
        # 顶部标题 & 描述
        gr.Markdown("""<center><font size=8>ChatBot</center>""")
        gr.Markdown(
            """<center><font size=3>This WebUI is based on GPT-style ChatBot, developed by Z.Y. Lin.</center>""")

        # Chatbot 组件 & 状态 & 文本输入
        chatbot = gr.Chatbot(label="ChatBot", height=400, type="messages")
        state = gr.State([])  # 用来存放 list of (q, r)
        txt = gr.Textbox(lines=2,
                         placeholder="请输入中文...",
                         label="Input")

        # 操作按钮
        with gr.Row():
            btn_clear = gr.Button("🧹 Clear History (清除历史)")
            btn_submit = gr.Button("🚀 Submit (发送)")
            btn_regen = gr.Button("🤔️ Regenerate (重试)")

        # Submit：发送新消息
        btn_submit.click(
            fn=lambda user_input, history: predict(user_input, history, model, proc, device),
            inputs=[txt, state],
            outputs=[chatbot, state]
        )
        
        # 回车也发消息
        txt.submit(
            fn=lambda user_input, history: predict(user_input, history, model, proc, device),
            inputs=[txt, state],
            outputs=[chatbot, state]
        )
        
        # 清空输入框
        btn_submit.click(lambda _: "", inputs=txt, outputs=txt)
        txt.submit(lambda _: "", inputs=txt, outputs=txt)

        # Regenerate：重新生成最后一条回复
        btn_regen.click(
            fn=lambda history: regenerate(history, model, proc, device),
            inputs=[state],
            outputs=[chatbot, state]
        )

        # Clear：清空历史
        btn_clear.click(
            fn=clear_history,
            inputs=None,
            outputs=[chatbot, state]
        )

        # 底部版权
        gr.HTML("""
        <div style="text-align:center; margin-top:1em;">
          <small>© 2025 Z.Y. Lin</small>
        </div>
        """)
    
    demo.launch(share=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chatbot.pt')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deploy(args.model, args.tokenizer, device)
