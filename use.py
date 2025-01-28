import torch
from torchvision.utils import save_image
from utils.image_generator import Generator
from utils.text_editor import encode_text


# Load generator
text_dim = 512
noise_dim = 100
img_channels = 3
img_size = 64
batch_size = 16
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
