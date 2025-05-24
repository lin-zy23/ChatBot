import argparse
import random

import torch
import gradio as gr

from call import load_model, generate_response


def predict(user_input: str,
            chat_history: list,
            model, proc, device):
    # chat_history + 新输入
    past = []
    for q, r in chat_history:
        past.extend([q, r])
    past.append(user_input)
    
    if user_input[-1] not in [',', '。', '！', '？']:
        user_input += random.choice(['。', '！', '？', ''])
    
    response = generate_response(model, proc, past, device)
    chat_history = chat_history + [(user_input, response)]
    
    return chat_history, chat_history

def regenerate(chat_history: list,
               model, proc, device):
    if not chat_history:
        return [], []
    last_q, _ = chat_history[-1]
    chat_history = chat_history[:-1]
    
    past = []
    for q, r in chat_history:
        past.extend([q, r])
    past.append(last_q)
    
    response = generate_response(model, proc, past, device)
    chat_history = chat_history + [(last_q, response)]
    
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
        chatbot = gr.Chatbot(label="ChatBot", height=400)
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