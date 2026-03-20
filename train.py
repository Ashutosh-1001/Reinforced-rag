import json
import torch
from pathlib import Path
from src.rag import ReinforcedRAG


def load_training_pairs(filepath):
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def train(
    data_path,
    pairs_path,
    epochs=3,
    checkpoint_dir="checkpoints",
    log_every=10,
):
    Path(checkpoint_dir).mkdir(exist_ok=True)

    rag = ReinforcedRAG(data_path)
    pairs = load_training_pairs(pairs_path)

    print(f"Training on {len(pairs)} query-answer pairs for {epochs} epoch(s).\n")

    for epoch in range(1, epochs + 1):
        epoch_losses, epoch_rewards = [], []

        for i, pair in enumerate(pairs):
            loss, reward = rag.train_on_query(pair["query"], pair["answer"])
            epoch_losses.append(loss)
            epoch_rewards.append(reward)

            if (i + 1) % log_every == 0:
                avg_loss = sum(epoch_losses[-log_every:]) / log_every
                avg_reward = sum(epoch_rewards[-log_every:]) / log_every
                print(
                    f"Epoch {epoch} | Step {i+1}/{len(pairs)} "
                    f"| Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}"
                )

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        print(f"\nEpoch {epoch} complete — avg loss: {avg_loss:.4f}, avg reward: {avg_reward:.4f}\n")

        ckpt_path = f"{checkpoint_dir}/policy_epoch{epoch}.pt"
        torch.save(rag.policy_net.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}\n")

    return rag


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    train(
        data_path=args.data,
        pairs_path=args.pairs,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
    )
