import os
import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    tokens_seen,
    train_losses,
    val_losses,
    track_tokens_seen,
):
    """Save everything needed to exactly resume training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, device):
    """Load a checkpoint and return the state needed to resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Checkpoint loaded from {path} (resuming at step {checkpoint['global_step']})"
    )
    return checkpoint
