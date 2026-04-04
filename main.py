from __future__ import annotations

import argparse

import torch

from models import build_svlb_from_config
from utils import SimpleTokenizer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an S-VLB smoke test with synthetic inputs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--text", type=str, default="a compact mobile robot moving through a warehouse")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_svlb_from_config(config)
    model.eval()

    image_size = config["data"]["image_size"]
    tokenizer = SimpleTokenizer(vocab_size=config["model"]["text"]["vocab_size"])
    encoded = tokenizer.batch_encode([args.text] * args.batch_size, max_length=config["model"]["text"]["max_length"])
    images = torch.randn(args.batch_size, 3, image_size, image_size)

    with torch.inference_mode():
        output = model(images=images, token_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])

    print("match_logit:", output.match_logit)
    print("vision_tokens:", tuple(output.vision_tokens.shape))
    print("text_tokens:", tuple(output.text_tokens.shape))
    print("spatial_size:", output.spatial_size)


if __name__ == "__main__":
    main()