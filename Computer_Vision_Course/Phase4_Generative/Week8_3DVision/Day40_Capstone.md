# Day 40: Final Capstone Project

## 1. Project Overview
**Goal:** Build a complete, deployable Generative AI application.
**Options:**
1.  **AI Art Generator:** A web app that takes text prompts and generates images using Stable Diffusion.
2.  **3D Asset Generator:** A tool that takes text/images and generates 3D models for games using Shap-E.
3.  **Visual Chatbot:** A chatbot that can see and discuss images using LLaVA/BLIP.

## 2. Option A: AI Art Generator (Stable Diffusion)
**Stack:** Python, PyTorch, Diffusers, Gradio, Hugging Face Spaces.

### Step 1: Pipeline Setup
```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate(prompt, steps, guidance):
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
    return image
```

### Step 2: UI with Gradio
```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# My AI Art Studio")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            steps = gr.Slider(10, 100, value=50, label="Steps")
            guidance = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
            btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Result")
            
    btn.click(generate, inputs=[prompt, steps, guidance], outputs=output)

demo.launch()
```

## 4. Option C: Visual Chatbot (Multimodal RAG)
**Goal:** Chat with your images. "What is in this picture?"
*   **Model:** LLaVA-1.5-7b (via 4-bit quantization) or BLIP-2.
*   **Stack:** Transformers, BitsAndBytes, Gradio.

### Step 1: Pipeline Setup
```python
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

# Load 4-bit quantized model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
)

def chat(image, prompt):
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0], skip_special_tokens=True)
```

### Step 2: Multimodal RAG (Advanced)
*   **Scenario:** You have 1000 PDFs with charts.
*   **Pipeline:**
    1.  **Extract:** Convert PDF pages to images.
    2.  **Embed:** Use CLIP to embed all images. Store in Vector DB (ChromaDB).
    3.  **Retrieve:** User asks "Show me the sales chart". Retrieve top-k images.
    4.  **Answer:** Pass retrieved image + question to LLaVA.
    5.  **Response:** "Here is the chart. It shows a 20% increase..."

## 5. Summary
This capstone demonstrates your ability to not just train models, but to **productize** them. You are now a Full-Stack AI Engineer.
