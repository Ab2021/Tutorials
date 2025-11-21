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

## 3. Option B: 3D Asset Generator (Shap-E)
**Goal:** Generate `.obj` or `.glb` files from text.
*   **Model:** OpenAI Shap-E.
*   **Pipeline:**
    1.  Text $\to$ Latent Representation (Transformer).
    2.  Latent $\to$ NeRF/SDF.
    3.  Marching Cubes $\to$ Mesh.
*   **Use Case:** Rapid prototyping for Indie Game Devs.

## Summary
This capstone demonstrates your ability to not just train models, but to **productize** them. You are now a Full-Stack AI Engineer.
