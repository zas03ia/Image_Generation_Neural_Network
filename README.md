# AI-Generated Logo Creation Using GAN

This project leverages a Generative Adversarial Network (GAN) to create unique and creative logos based on text prompts. By combining a generator and discriminator, the model generates high-quality logos using input text descriptions, which are encoded into embeddings using CLIP (Contrastive Language-Image Pretraining).

---

## Overview

The goal of this project is to train a model that can generate logos for different concepts and companies based on textual descriptions. The system is built on a GAN architecture, where the **Generator** creates the logos from the input text embeddings and random noise, and the **Discriminator** evaluates how realistic the generated logos are.

### Key Components:

1. **Generator**: This model takes text embeddings and noise as input to generate images (logos).
2. **Discriminator**: This model evaluates real and fake images and returns a validity score.
3. **Text Embedding**: The text descriptions are encoded using a CLIP-based encoder model, which transforms the text into embeddings.
4. **Training Loop**: The model is trained using a combination of real and fake logo images, adjusting the weights of the generator and discriminator based on loss functions.

---

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/zas03ia/Image_Generation_Neural_Network.git
cd AI-Logo-Generation
```

### Dependencies

Install them using `pip`:

```bash
pip install -r requirements.txt
```

---

## Model Details

### Generator

The generator is a fully connected neural network followed by several transposed convolution layers to upscale the image to the desired size. It takes both text embeddings (from the CLIP model) and random noise as input to generate images.

```python
class Generator(nn.Module):
    def __init__(self, text_dim, noise_dim, img_channels, img_size):
        ...
```

### Discriminator

The discriminator uses convolution layers followed by fully connected layers to differentiate between real and fake logos. It also takes the text embedding as an additional input to ensure the generated logos match the provided description.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size, text_dim):
        ...
```

### Text Encoder

We use the CLIP model, a powerful multimodal model, to encode text descriptions into embeddings. This ensures that the generated logos correspond to the semantic meaning of the text prompt.

```python
def encode_text(prompt):
    ...
```

---

## Training

To train the model, use the following parameters:

- **Learning rate**: 0.0002
- **Batch size**: 16
- **Image size**: 64x64
- **Epochs**: 100

The training loop optimizes both the **Generator** and **Discriminator** models using the **BCEWithLogitsLoss** loss function, which is ideal for binary classification problems.

### Training Script

```python
for epoch in range(epochs):
    for i, text_prompt in enumerate(["A minimalist logo for a tech company"] * batch_size):
        ...
```

The model saves generated logos every 10 epochs as PNG files.

---

## Usage

After training, the generator model can be used to create logos from text prompts.

### Example:

```python
# Load generator
generator = Generator(text_dim, noise_dim, img_channels, img_size)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

# Text prompt
text_prompt = "A creative logo for a futuristic AI company"
text_embedding = encode_text(text_prompt).mean(dim=1).to("cpu")

# Generate logo
noise = torch.randn(batch_size, noise_dim).to("cpu")
with torch.no_grad():
    fake_img = generator(text_embedding, noise)

# Save logo
save_image(fake_img, "generated_logo.png", normalize=True)
print("Generated logo saved as generated_logo.png")
```

The generated logo will be saved as `generated_logo.png`.

---

## Future Improvements

- **Incorporating Style Transfer**: Using pre-trained models to incorporate specific styles into the generated logos.
- **Better Text-to-Image Synthesis**: Experiment with more advanced models like DALL·E for higher-quality logo generation.
- **User Interface**: Develop a web interface where users can input text prompts and get logos in real-time.

---

## Acknowledgments

- The **CLIP** model by OpenAI for text-to-image embeddings.
- The **PyTorch** framework for deep learning development.
- **Torchvision** for image utilities like saving and transforming images.

