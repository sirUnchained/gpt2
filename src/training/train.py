import os
import math

import torch
import tiktoken

from scripts.evaluate import generate_text, text_to_token_ids, token_ids_to_text
from configs.model_configs import GPT_configs
from src.data.dataset import create_dataloader
from src.training.loss import calc_loader_cost, calc_batch_cost
from src.models.gpt_model import GPT_model
from src.data.dataset import load_text_data
from src.utils.ckeckpoints import load_checkpoint, save_checkpoint


def train_model(
    model,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_path: str,
    checkpoint_freq: int = 1000,  # save every N global steps, in addition to per-epoch
    use_checkpoints=False,
    create_checkpoints=False,
):
    """
    Trains a language model over multiple epochs with periodic evaluation and checkpointing.

    The training loop iterates over batches from `train_dataloader`, computes the loss via
    `calc_batch_cost`, backpropagates, and updates the model weights. At intervals defined
    by `eval_freq` (global steps), the model is evaluated on the full training and validation
    datasets, and the losses are recorded. After each complete epoch, a text sample is
    generated using `start_context` to monitor qualitative performance.

    Checkpointing is managed by the boolean parameters `use_checkpoints` and
    `create_checkpoints`. When resuming (`use_checkpoints=True`), the function looks for
    `latest.pt` in `checkpoint_path`. When saving (`create_checkpoints=True`), it writes
    periodic snapshots (`latest.pt` and epoch-named files) to the same directory.

    Args:
        model (torch.nn.Module): The language model to be trained.
        train_dataloader (DataLoader): DataLoader yielding training batches.
        val_dataloader (DataLoader): DataLoader yielding validation batches.
        num_epochs (int): Number of complete passes over the training data.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient-based updates.
        device (torch.device): Device (CPU or CUDA) on which to perform computations.
        eval_freq (int): Evaluate the model every N global steps.
        eval_iter (int): (Unused) Placeholder for the number of batches to use during
                         evaluation. Currently, `evaluate_model` ignores this value.
        start_context (str): Initial text prompt used for sample generation after each epoch.
        tokenizer: Tokenizer instance for converting text to token IDs and vice versa.
        checkpoint_path (str): Directory where checkpoint files are read from and written to.
        checkpoint_freq (int, optional): If `create_checkpoints` is True, save a checkpoint
                                         every N global steps. Defaults to 1000.
        use_checkpoints (bool, optional): If True, attempt to resume training from
                                          `{checkpoint_path}/latest.pt`. Defaults to False.
        create_checkpoints (bool, optional): If True, save periodic and end-of-epoch
                                              checkpoints to `checkpoint_path`. Defaults to False.

    Returns:
        tuple: A 3-element tuple containing:
            - train_losses (list): Recorded average training losses at each evaluation step.
            - val_losses (list): Recorded average validation losses at each evaluation step.
            - track_tokens_seen (list): Cumulative number of tokens processed at each
                                        evaluation step, used for plotting loss vs. tokens.

    Note:
        The learning rate is dynamically adjusted inside the loop via
        `learning_rate_change`, which is expected to be defined in the outer scope.
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    total_steps = len(train_dataloader) * num_epochs
    start_epoch = 0

    latest_ckpt_path = os.path.join(checkpoint_path, "latest.pt")

    # --- Resume from checkpoint if requested and available ---
    if use_checkpoints and os.path.exists(latest_ckpt_path):
        checkpoint = load_checkpoint(latest_ckpt_path, model, optimizer, device)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        tokens_seen = checkpoint["tokens_seen"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        track_tokens_seen = checkpoint["track_tokens_seen"]
    elif use_checkpoints:
        print(
            f"`use_checkpoints` is true but no checkpoint found at {latest_ckpt_path}, starting fresh."
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_batch_cost(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            learning_rate_change(global_step, total_steps, 0.2, optimizer)

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_losses.append(train_loss.item())  # type: ignore
                val_losses.append(val_loss.item())  # type: ignore

                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

            # --- Periodic checkpoint save ---
            if (
                create_checkpoints
                and global_step % checkpoint_freq == 0
                and global_step > 0
            ):
                save_checkpoint(
                    latest_ckpt_path,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    tokens_seen,
                    train_losses,
                    val_losses,
                    track_tokens_seen,
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

        # --- End-of-epoch checkpoint save ---
        if create_checkpoints:
            epoch_ckpt_path = os.path.join(checkpoint_path, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                epoch_ckpt_path,
                model,
                optimizer,
                epoch + 1,
                global_step,
                tokens_seen,
                train_losses,
                val_losses,
                track_tokens_seen,
            )
            # also update "latest" so resuming picks up here
            save_checkpoint(
                latest_ckpt_path,
                model,
                optimizer,
                epoch + 1,
                global_step,
                tokens_seen,
                train_losses,
                val_losses,
                track_tokens_seen,
            )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    """
    ## Evaluate the model on the training and validation dataloaders.

    This function computes the average loss over all batches in the training
    and validation sets using `calc_loader_cost`. The model is temporarily
    set to evaluation mode, and then restored to training mode.

    ---

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device on which the tensors are allocated.
        eval_iter (int): Number of batches to use for evaluation (currently unused,
                         kept for compatibility with the training loop).

    Returns:
        tuple: (train_loss, val_loss) where each is a scalar tensor representing
               the average cross-entropy loss over the respective dataloader.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loader_cost(train_dataloader, model, device, None)
        val_loss = calc_loader_cost(val_dataloader, model, device, None)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    ## Generate a text sample from the model and print it.

    Useful for monitoring training progress: after each epoch, this function
    generates a fixed number of tokens (10) conditioned on `start_context`
    and prints the resulting text on a single line.

    ---

    Args:
        model (torch.nn.Module): The language model used for generation.
        tokenizer: Tokenizer object that converts between text and token ids.
        device (torch.device): Device where the model and tensors reside.
        start_context (str): Initial prompt string to condition the generation.

    Returns:
        None
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded_text = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, encoded_text, 10, context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Print sample as a single line
    model.train()


def learning_rate_change(
    global_step: int,
    total_training_steps: int,
    warmup_percent: float,
    optimizer: torch.optim.Optimizer,
    initial_lr=1e-5,
    peak_lr=3e-4,
) -> float:
    """
    ## Update the optimizer's learning rate using a warmup then cosine decay schedule.

    During the warmup phase (first `warmup_percent` fraction of total steps), the learning rate increases linearly from `initial_lr` to `peak_lr`.
    After warmup, the learning rate decays following a cosine curve from `peak_lr` down to `min_lr = 0.1 * initial_lr` over the remaining steps.

    Args:
        global_step (int): Current training step (0‑based).
        total_training_steps (int): Total number of training steps.
        warmup_percent (float): Fraction of total steps used for warmup (e.g., 0.1 for 10%).
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be updated.
        initial_lr (float, optional): Starting learning rate before warmup. Default 1e-5.
        peak_lr (float, optional): Maximum learning rate reached at the end of warmup. Default 3e-4.

    Returns:
        float: The updated learning rate value (from the first parameter group).
    """

    warmup_steps = int(warmup_percent * total_training_steps)
    lr_increment = (peak_lr - initial_lr) / warmup_steps if warmup_steps > 0 else 0
    min_lr = 0.1 * initial_lr

    if global_step < warmup_steps:
        # If the global step is less than warmup steps so we increase lr
        lr = initial_lr + global_step * lr_increment
    else:
        # If the global step is greater or equal to warmup steps so we start cosine decay
        progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
        progress = min(progress, 1.0)  # guard against overshoot
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos((math.pi * progress)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer.param_groups[0]["lr"]


if __name__ == "__main__":
    cfg = GPT_configs()

    text = load_text_data(cfg)
    train_ratio = 0.80
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]

    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)
    train_loader = create_dataloader(train_data)
    val_loader = create_dataloader(val_data)
    epochs = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(123)
    model = GPT_model(cfg)
    model.to(device=device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=5e-4, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        epochs,
        optim,
        device,
        eval_freq=5,
        eval_iter=5,
        start_context="Hello I am ",
        tokenizer=tokenizer,
        checkpoint_path=cfg.checkpoints_path,
    )
