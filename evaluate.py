from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

from models import build_svlb_from_config
from utils import MultilingualTokenizer, build_image_transform, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Thai retrieval with Recall@K.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def batched(items: list[object], batch_size: int) -> list[list[object]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def compute_recall(similarity: torch.Tensor, caption_to_image: list[int], ks: tuple[int, ...]) -> dict[str, float]:
    image_count = similarity.size(0)
    max_k = min(max(ks), similarity.size(1), similarity.size(0))
    image_hits = {k: 0 for k in ks}
    text_hits = {k: 0 for k in ks}

    positives_per_image: dict[int, set[int]] = {index: set() for index in range(image_count)}
    for caption_index, image_index in enumerate(caption_to_image):
        positives_per_image[image_index].add(caption_index)

    topk_text = similarity.topk(k=min(max_k, similarity.size(1)), dim=1).indices
    for image_index in range(image_count):
        ranked = topk_text[image_index].tolist()
        positives = positives_per_image[image_index]
        for k in ks:
            if any(candidate in positives for candidate in ranked[: min(k, len(ranked))]):
                image_hits[k] += 1

    topk_image = similarity.topk(k=min(max_k, similarity.size(0)), dim=0).indices.transpose(0, 1)
    for caption_index, image_index in enumerate(caption_to_image):
        ranked = topk_image[caption_index].tolist()
        for k in ks:
            if image_index in ranked[: min(k, len(ranked))]:
                text_hits[k] += 1

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"image_to_text_r@{k}"] = image_hits[k] / max(image_count, 1)
        metrics[f"text_to_image_r@{k}"] = text_hits[k] / max(len(caption_to_image), 1)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device or config["inference"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    tokenizer = MultilingualTokenizer(
        model_name=config["model"]["text"]["pretrained_model_name"],
        max_length=config["model"]["text"]["max_length"],
        use_fast=config["model"]["text"].get("tokenizer_use_fast", True),
    )
    transform = build_image_transform(config["data"]["image_size"])
    model = build_svlb_from_config(config).to(device)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    stream = iter(load_dataset(config["train"]["dataset_name"], split=args.split, streaming=True))
    samples: list[dict[str, object]] = []
    while len(samples) < args.num_images:
        example = next(stream)
        image = example["image"].convert("RGB")
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image from the Hugging Face dataset stream.")
        captions = [str(caption) for caption in example["th_sentences_raw"] if str(caption).strip()]
        if not captions:
            continue
        samples.append({"image": image, "captions": captions})

    model.eval()
    image_embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        image_tensors = [transform(sample["image"]) for sample in samples]
        for image_batch in batched(image_tensors, args.batch_size):
            images = torch.stack(image_batch, dim=0).to(device)
            embeddings, _, _ = model.encode_image_embedding(images)
            image_embeddings.append(embeddings.cpu())
    image_embedding_tensor = torch.cat(image_embeddings, dim=0)

    all_captions: list[str] = []
    caption_to_image: list[int] = []
    for image_index, sample in enumerate(samples):
        for caption in sample["captions"]:
            all_captions.append(caption)
            caption_to_image.append(image_index)

    text_embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for caption_batch in batched(all_captions, args.batch_size):
            encoded = tokenizer.batch_encode(caption_batch)
            embeddings, _ = model.encode_text_embedding(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            text_embeddings.append(embeddings.cpu())
    text_embedding_tensor = torch.cat(text_embeddings, dim=0)

    similarity = model.logit_scale.exp().detach().cpu() * image_embedding_tensor @ text_embedding_tensor.transpose(0, 1)
    metrics = compute_recall(similarity, caption_to_image, ks=(1, 5, 10))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}={metric_value:.4f}")


if __name__ == "__main__":
    main()