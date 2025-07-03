from sentence_transformers import SentenceTransformer
import torch
import time
import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np # Import numpy

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
original_model_path = "./original_embedder_model"
onnx_model_path = "./onnx_embedder_model"
quantized_onnx_model_path = "./quantized_embedder_model.onnx"

# 1. Load the Original Embedder Model
print("Loading original embedder model...")
embedder_model = SentenceTransformer(model_name)
embedder_model.to('cpu')
embedder_model.eval()
print("Original embedder model loaded.")

# Save the original model to get its size
embedder_model.save(original_model_path)
original_model_size = sum(os.path.getsize(os.path.join(original_model_path, f)) for f in os.listdir(original_model_path) if os.path.isfile(os.path.join(original_model_path, f))) / (1024 * 1024)
print(f"Original model size: {original_model_size:.2f} MB")

# 2. Measure Baseline Performance (Speed)
texts_to_embed = ["This is a test sentence.", "Another sentence to embed."] * 200

print("Measuring baseline embedding time...")
_ = embedder_model.encode(texts_to_embed[:10], convert_to_tensor=False) # Warm-up
start_time = time.time()
embeddings = embedder_model.encode(texts_to_embed, convert_to_tensor=False)
end_time = time.time()
baseline_embedding_time = end_time - start_time
print(f"Baseline embedding time: {baseline_embedding_time:.4f} seconds")

# 3. Export to ONNX
print("Exporting model to ONNX...")
# Load the model and export it to ONNX format
ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
ort_model.save_pretrained(onnx_model_path)
print(f"Model exported to ONNX at: {onnx_model_path}")

# 4. Apply ONNX Runtime Dynamic Quantization
print("Applying ONNX Runtime dynamic quantization...")
# The quantize_dynamic function expects a path to the ONNX model file, not a directory.
# So, we need to specify the actual ONNX model file inside the exported directory.
# By default, optimum saves the model as 'model.onnx' inside the specified output_path.
quantize_dynamic(
    os.path.join(onnx_model_path, "model.onnx"),
    quantized_onnx_model_path,
    op_types_to_quantize=['MatMul', 'Add'],
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8
)
print(f"Quantized ONNX model saved at: {quantized_onnx_model_path}")

# 5. Measure Quantized Performance (Size & Speed)
quantized_model_size = os.path.getsize(quantized_onnx_model_path) / (1024 * 1024)
print(f"Quantized model size: {quantized_model_size:.2f} MB")

print("Measuring quantized model embedding time...")
# To measure speed, we need to load the quantized ONNX model and use onnxruntime for inference
import onnxruntime as rt

session = rt.InferenceSession(quantized_onnx_model_path)
input_names = [inp.name for inp in session.get_inputs()]
output_name = session.get_outputs()[0].name

# Prepare inputs for ONNX Runtime
# This part is a bit more involved as SentenceTransformer handles tokenization internally.
# We need to replicate the tokenization process.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Warm-up for quantized model
tokenized_inputs = tokenizer(texts_to_embed[:10], padding=True, truncation=True, return_tensors="np")
_ = session.run(
    [output_name],
    {
        "input_ids": tokenized_inputs['input_ids'].astype(np.int64),
        "attention_mask": tokenized_inputs['attention_mask'].astype(np.int64)
    }
)

start_time = time.time()
tokenized_inputs = tokenizer(texts_to_embed, padding=True, truncation=True, return_tensors="np")
# ONNX Runtime expects numpy arrays
onnx_embeddings = session.run(
    [output_name],
    {
        "input_ids": tokenized_inputs['input_ids'].astype(np.int64),
        "attention_mask": tokenized_inputs['attention_mask'].astype(np.int64)
    }
)
end_time = time.time()
quantized_embedding_time = end_time - start_time
print(f"Quantized Embedding Time: {quantized_embedding_time:.4f} seconds")

print("\n--- Performance Comparison ---")
print(f"Original Model Size: {original_model_size:.2f} MB")
print(f"Quantized Model Size: {quantized_model_size:.2f} MB (Reduction: {((original_model_size - quantized_model_size) / original_model_size) * 100:.2f}%) ")
print(f"Baseline Embedding Time: {baseline_embedding_time:.4f} seconds")
print(f"Quantized Embedding Time: {quantized_embedding_time:.4f} seconds (Speedup: {((baseline_embedding_time - quantized_embedding_time) / baseline_embedding_time) * 100:.2f}%) ")

# Clean up original model directory
import shutil
if os.path.exists(original_model_path):
    shutil.rmtree(original_model_path)

# Clean up onnx_model_path directory
if os.path.exists(onnx_model_path):
    shutil.rmtree(onnx_model_path)
