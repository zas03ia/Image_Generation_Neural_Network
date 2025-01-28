import torch
from transformers import CLIPTextModel, CLIPTokenizer

# Load CLIP tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
batch_size = 16

def encode_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_features = text_encoder(**inputs).last_hidden_state
    return text_features.expand(batch_size, -1, -1)  # Shape: [batch_size, sequence_length, embedding_dim]
