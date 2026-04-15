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
from utils import MultilingualTokenizer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S-VLB (Text-only) with a streaming Thai MS-COCO dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


class TextCollateFn:
    def __init__(self, tokenizer: MultilingualTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Tensor | list[str]]:
        captions = [str(item["caption"]) for item in batch]
        encoded = self.tokenizer.batch_encode(captions, max_length=self.max_length)
        return {
            "captions": captions,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


def compute_lm_loss(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: torch.device,
) -> Tensor:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Standard Next Token Prediction implementation using shifted inputs
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, :-1]

    logits, _ = model(token_ids=inputs, attention_mask=mask)
    
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.contiguous().view(-1)
    
    loss_unreduced = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    target_mask = attention_mask[:, 1:].contiguous().view(-1).float()
    loss = (loss_unreduced * target_mask).sum() / target_mask.sum().clamp_min(1.0)
    
    return loss


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

    tokenizer = MultilingualTokenizer(
        model_name=config["model"]["text"]["pretrained_model_name"],
        max_length=config["model"]["max_length"],
        use_fast=config["model"]["text"].get("tokenizer_use_fast", True),
    )
    
    dataset = ThaiCOCODataset(
        split=train_config["split"],
        dataset_name=train_config["dataset_name"],
        use_all_captions=train_config.get("use_all_captions", True),
        shuffle_buffer_size=train_config.get("shuffle_buffer_size"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=TextCollateFn(tokenizer, config["model"]["max_length"]),
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

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_lm_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=device,
            )
            loss.backward()
            clip_norm = train_config.get("gradient_clip_norm")
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            step_losses.append(float(loss.item()))

            if step % 10 == 0 or step == 1:
                print(
                    " ".join(
                        [
                            f"epoch={epoch}",
                            f"step={step}",
                            f"loss={running_loss / step:.4f}",
                        ]
                    )
                )

            if step >= max_steps_per_epoch:
                break

        epoch_loss = running_loss / step
        epoch_losses.append(epoch_loss)

        print(
            " ".join(
                [
                    f"epoch={epoch}",
                    "done",
                    f"loss={epoch_loss:.4f}",
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
        writer.writerow(["epoch", "loss"])
        for epoch_idx, loss_value in enumerate(epoch_losses, start=1):
            writer.writerow([epoch_idx, loss_value])
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
        axes[1].set_title("Epoch Curves")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss")
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