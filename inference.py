from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from models import build_svlb_from_config
from utils import SimpleTokenizer, build_image_transform, load_config, sigmoid_confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot S-VLB inference on an image and candidate texts.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", action="append", required=True, dest="texts")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device or config["inference"].get("device", "cpu")
    device = torch.device(device_name)
    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}. Replace --image with a real file path, for example --image ./samples/cat.jpg"
        )

    tokenizer = SimpleTokenizer(vocab_size=config["model"]["text"]["vocab_size"])
    transform = build_image_transform(config["data"]["image_size"])
    model = build_svlb_from_config(config).to(device)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError(f"Unsupported or corrupted image file: {image_path}") from error
    image_tensor = transform(image).unsqueeze(0).to(device)
    batch_size = len(args.texts)
    image_batch = image_tensor.expand(batch_size, -1, -1, -1)
    encoded = tokenizer.batch_encode(args.texts, max_length=config["model"]["text"]["max_length"])
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        output = model(images=image_batch, token_ids=input_ids, attention_mask=attention_mask)
        probabilities = sigmoid_confidence(output.match_logit)

    ranked = sorted(zip(args.texts, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    for candidate_text, score in ranked:
        print(f"{score:.4f}\t{candidate_text}")


if __name__ == "__main__":
    main()