# Lab 4: Quantization Script (GGUF)

## Objective
Run LLMs on CPU/Mac.
Convert a model to **GGUF** format.

## 1. Setup
Clone `llama.cpp`.

## 2. The Command
```bash
# 1. Download Model
huggingface-cli download microsoft/phi-2 --local-dir phi-2

# 2. Convert to FP16 GGUF
python llama.cpp/convert.py phi-2 --outtype f16 --outfile phi-2-f16.gguf

# 3. Quantize to Q4_K_M (4-bit)
./llama.cpp/quantize phi-2-f16.gguf phi-2-q4.gguf q4_k_m
```

## 3. Analysis
Compare file sizes.
FP16: ~5GB.
Q4: ~1.5GB.

## 4. Submission
Submit the file size of the Q4 GGUF file.
