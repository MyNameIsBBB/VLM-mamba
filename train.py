from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import ThaiCOCODataset
from models import build_svlb_from_config
from utils import MultilingualTokenizer, build_image_transform, load_config, retrieval_topk_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S-VLB with a streaming Thai MS-COCO dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def build_collate_fn(tokenizer: MultilingualTokenizer, max_length: int):
    def collate_fn(batch: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
        images = torch.stack([item["image"] for item in batch], dim=0)
        captions = [str(item["caption"]) for item in batch]
        image_ids = torch.tensor([int(item["image_id"]) for item in batch], dtype=torch.long)
        encoded = tokenizer.batch_encode(captions, max_length=max_length)
        return {
            "images": images,
            "captions": captions,
            "image_ids": image_ids,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return collate_fn


def compute_contrastive_loss(
    model: nn.Module,
    images: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    image_ids: Tensor,
    device: torch.device,
    hard_negative_margin: float,
    hard_negative_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    image_ids = image_ids.to(device)

    output = model(images=images, token_ids=input_ids, attention_mask=attention_mask)
    similarity = output.similarity_matrix
    positive_mask = image_ids.unsqueeze(1) == image_ids.unsqueeze(0)

    image_targets = positive_mask.float()
    image_targets = image_targets / image_targets.sum(dim=1, keepdim=True).clamp_min(1.0)
    text_targets = image_targets.transpose(0, 1)

    image_loss = -(image_targets * F.log_softmax(similarity, dim=1)).sum(dim=1).mean()
    text_loss = -(text_targets * F.log_softmax(similarity.transpose(0, 1), dim=1)).sum(dim=1).mean()
    loss = 0.5 * (image_loss + text_loss)

    if hard_negative_weight > 0 and similarity.size(0) > 1:
        positive_scores = (similarity * image_targets).sum(dim=1)
        hard_negative_scores = similarity.masked_fill(positive_mask, float("-inf")).max(dim=1).values
        valid_rows = torch.isfinite(hard_negative_scores)
        if valid_rows.any():
            hard_negative_loss = F.relu(
                hard_negative_margin + hard_negative_scores[valid_rows] - positive_scores[valid_rows]
            ).mean()
            loss = loss + hard_negative_weight * hard_negative_loss

    return loss, similarity.detach(), positive_mask.detach()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config["train"]
    device_name = (
        args.device
        or train_config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_name)

    batch_size = args.batch_size or train_config["batch_size"]
    epochs = args.epochs or train_config["epochs"]
    learning_rate = args.lr or train_config["lr"]
    num_workers = args.num_workers if args.num_workers is not None else train_config["num_workers"]
    max_steps_per_epoch = args.max_steps_per_epoch or train_config["max_steps_per_epoch"]
    if args.temperature is not None:
        train_config["temperature"] = args.temperature
    tokenizer = MultilingualTokenizer(
        model_name=config["model"]["text"]["pretrained_model_name"],
        max_length=config["model"]["text"]["max_length"],
        use_fast=config["model"]["text"].get("tokenizer_use_fast", True),
    )
    image_transform = build_image_transform(config["data"]["image_size"])
    dataset = ThaiCOCODataset(
        split=train_config["split"],
        transform=image_transform,
        dataset_name=train_config["dataset_name"],
        use_all_captions=train_config.get("use_all_captions", True),
        shuffle_buffer_size=train_config.get("shuffle_buffer_size"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer, config["model"]["text"]["max_length"]),
    )

    model = build_svlb_from_config(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=train_config["weight_decay"])
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plot_path = Path(args.plot_path) if args.plot_path else checkpoint_dir / "train_curves.png"
    history_path = Path(args.history_path) if args.history_path else checkpoint_dir / "train_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    epoch_accuracies: list[float] = []

    start_epoch = 1
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f">>> Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            model.load_state_dict(checkpoint)
            if "epoch_" in args.checkpoint:
                try:
                    start_epoch = int(args.checkpoint.split("epoch_")[-1].split(".")[0]) + 1
                except ValueError:
                    pass

    if start_epoch > epochs:
        print(f">>> Checkpoint is already at epoch {start_epoch - 1}, nothing to train for epochs={epochs}.")
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        running_image_r1 = 0.0
        running_text_r1 = 0.0

        for step, batch in enumerate(dataloader, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_ids = batch["image_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, similarity, positive_mask = compute_contrastive_loss(
                model=model,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=image_ids,
                device=device,
                hard_negative_margin=train_config.get("hard_negative_margin", 0.2),
                hard_negative_weight=train_config.get("hard_negative_weight", 0.0),
            )
            loss.backward()
            clip_norm = train_config.get("gradient_clip_norm")
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            running_image_r1 += float(retrieval_topk_accuracy(similarity, positive_mask, k=1, dim=1).item())
            running_text_r1 += float(retrieval_topk_accuracy(similarity, positive_mask, k=1, dim=0).item())
            step_losses.append(float(loss.item()))

            if step % 10 == 0 or step == 1:
                print(
                    " ".join(
                        [
                            f"epoch={epoch}",
                            f"step={step}",
                            f"loss={running_loss / step:.4f}",
                            f"image_r@1={running_image_r1 / step:.4f}",
                            f"text_r@1={running_text_r1 / step:.4f}",
                        ]
                    )
                )

            if step >= max_steps_per_epoch:
                break

        epoch_loss = running_loss / step
        epoch_accuracy = 0.5 * ((running_image_r1 / step) + (running_text_r1 / step))
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(
            " ".join(
                [
                    f"epoch={epoch}",
                    "done",
                    f"loss={epoch_loss:.4f}",
                    f"image_r@1={running_image_r1 / step:.4f}",
                    f"text_r@1={running_text_r1 / step:.4f}",
                ]
            )
        )

        checkpoint_path = checkpoint_dir / f"svlb_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            checkpoint_path,
        )
        print(f"saved checkpoint: {checkpoint_path}")

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "retrieval_r1"])
        for epoch_idx, (loss_value, accuracy_value) in enumerate(zip(epoch_losses, epoch_accuracies), start=1):
            writer.writerow([epoch_idx, loss_value, accuracy_value])
    print(f"saved history: {history_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(range(1, len(step_losses) + 1), step_losses, label="train loss", color="tab:red")
        axes[0].set_title("Step Loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.2)

        epochs_axis = list(range(1, len(epoch_losses) + 1))
        axes[1].plot(epochs_axis, epoch_losses, label="epoch loss", color="tab:blue")
        axes[1].plot(epochs_axis, epoch_accuracies, label="epoch accuracy", color="tab:green")
        axes[1].set_title("Epoch Curves")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("value")
        axes[1].grid(alpha=0.2)
        axes[1].legend()

        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"saved curves: {plot_path}")
    except ImportError:
        print("matplotlib is not installed; skipped curve plotting.")


if __name__ == "__main__":
    main()