import os
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "assets" / "generator.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at import time so each request is fast.
MODEL = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
MODEL.eval()

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    # Model output is in [-1, 1], convert to [0, 255].
    array = tensor.detach().cpu().clamp(-1, 1)
    array = ((array + 1) / 2.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def generate_image_from_sketch_bytes(input_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_tensor = MODEL(input_tensor)[0]

    output_image = tensor_to_rgb_image(output_tensor)
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()
