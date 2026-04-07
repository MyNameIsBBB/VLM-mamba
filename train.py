from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import ThaiCOCODataset
from models import build_svlb_from_config
from utils import SimpleTokenizer, binary_accuracy, build_image_transform, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S-VLB with a streaming Thai MS-COCO dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_collate_fn(tokenizer: SimpleTokenizer, max_length: int):
    def collate_fn(batch: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
        images = torch.stack([item["image"] for item in batch], dim=0)
        captions = [str(item["caption"]) for item in batch]
        encoded = tokenizer.batch_encode(captions, max_length=max_length)
        return {
            "images": images,
            "captions": captions,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return collate_fn


def compute_matching_loss(
    model: nn.Module,
    images: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    positive_logits = model(images=images, token_ids=input_ids, attention_mask=attention_mask).match_logit
    
    if images.size(0) > 1:
        negative_input_ids = torch.roll(input_ids, shifts=1, dims=0).to(device)
        negative_attention_mask = torch.roll(attention_mask, shifts=1, dims=0).to(device)
        
        negative_logits = model(
            images=images,
            token_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
        ).match_logit
        
        logits = torch.cat([positive_logits, negative_logits], dim=0)
        labels = torch.cat([
            torch.ones_like(positive_logits), 
            torch.zeros_like(negative_logits)
        ], dim=0).to(device) 
    else:
        logits = positive_logits
        labels = torch.ones_like(positive_logits).to(device)

    loss = criterion(logits, labels)
    return loss, logits.detach(), labels.detach()
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

    tokenizer = SimpleTokenizer(vocab_size=config["model"]["text"]["vocab_size"])
    image_transform = build_image_transform(config["data"]["image_size"])
    dataset = ThaiCOCODataset(
        split=train_config["split"],
        transform=image_transform,
        dataset_name=train_config["dataset_name"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer, config["model"]["text"]["max_length"]),
    )

    model = build_svlb_from_config(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=train_config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for step, batch in enumerate(dataloader, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, logits, labels = compute_matching_loss(
                model=model,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                criterion=criterion,
                device=device
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_accuracy += float(binary_accuracy(logits, labels).item())

            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch} step={step} loss={running_loss / step:.4f} accuracy={running_accuracy / step:.4f}"
                )

            if step >= max_steps_per_epoch:
                break

        checkpoint_path = checkpoint_dir / f"svlb_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()