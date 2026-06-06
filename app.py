import gradio as gr
import torch
from unsloth import FastLanguageModel

model_name = "revenue_growth_copilot_lora"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def predict(message, history):
    prompt = f"<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.4)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]
    return response

gr.ChatInterface(predict).launch()
