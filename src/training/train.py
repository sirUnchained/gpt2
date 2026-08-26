import os
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.evaluate import (
    generate_text,
    text_to_token_ids,
    token_ids_to_text,
)

from src.training.loss import (
    calc_batch_cost,
)

from src.utils.ckeckpoints import (
    load_checkpoint,
    save_checkpoint,
)

# ============================================================
# Distributed utilities
# ============================================================


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_distributed():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    """
    Return the underlying model when using DDP.
    """

    if isinstance(model, DDP):
        return model.module

    return model


# ============================================================
# Training
# ============================================================


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    optimizer,
    device,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_path,
    checkpoint_freq=1000,
    use_checkpoints=False,
    create_checkpoints=False,
    distributed=False,
):
    """
    Trains a language model over multiple epochs with periodic evaluation and checkpointing.

    The training loop iterates over batches from `train_dataloader`, computes the loss via
    `calc_batch_cost`, backpropagates, and updates the model weights. At intervals defined
    by `eval_freq` (global steps), the model is evaluated on the `eval_iter` batch size of training and validation
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
        eval_iter (int): Placeholder for the number of batches to use during
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
        distributed (bool): If True, we'll use distributed gpus.

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

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1
    start_epoch = 0

    total_steps = len(train_dataloader) * num_epochs

    latest_ckpt_path = os.path.join(
        checkpoint_path,
        "latest.pt",
    )

    # --------------------------------------------------------
    # Resume from checkpoint if requested and available
    # --------------------------------------------------------

    if use_checkpoints and os.path.exists(latest_ckpt_path):
        if is_main_process():
            print(f"Loading checkpoint: " f"{latest_ckpt_path}")

        # Load checkpoint into the raw model.
        # This avoids DDP's "module." state_dict issue.
        raw_model = unwrap_model(model)

        checkpoint = load_checkpoint(
            latest_ckpt_path,
            raw_model,
            optimizer,
            device,
        )

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        tokens_seen = checkpoint["tokens_seen"]

        train_losses = checkpoint["train_losses"]

        val_losses = checkpoint["val_losses"]

        track_tokens_seen = checkpoint["track_tokens_seen"]

        if is_main_process():
            print(
                f"Resuming from epoch "
                f"{start_epoch}, "
                f"global step "
                f"{global_step}"
            )

    elif use_checkpoints and is_main_process():

        print(
            "Checkpoint requested but no "
            "checkpoint was found. "
            "Starting from scratch."
        )

    # --------------------------------------------------------
    # Wrap model in DDP
    # --------------------------------------------------------

    if distributed:

        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
        )

    # --------------------------------------------------------
    # Synchronize all ranks
    # --------------------------------------------------------

    if is_distributed():
        dist.barrier()

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        num_epochs,
    ):

        model.train()

        # IMPORTANT:
        # DistributedSampler needs a new seed every epoch.
        if hasattr(
            train_dataloader,
            "sampler",
        ) and hasattr(
            train_dataloader.sampler,
            "set_epoch",
        ):

            train_dataloader.sampler.set_epoch(epoch)

        if hasattr(
            val_dataloader,
            "sampler",
        ) and hasattr(
            val_dataloader.sampler,
            "set_epoch",
        ):

            val_dataloader.sampler.set_epoch(epoch)

        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for input_batch, target_batch in train_dataloader:

            optimizer.zero_grad(set_to_none=True)

            loss = calc_batch_cost(
                input_batch,
                target_batch,
                model,
                device,
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Counters
            # ------------------------------------------------

            # In DDP, every rank sees batch_size samples.
            # We count global tokens only once.
            local_tokens = input_batch.numel()

            if distributed:
                tokens_tensor = torch.tensor(
                    local_tokens,
                    device=device,
                    dtype=torch.long,
                )

                dist.all_reduce(
                    tokens_tensor,
                    op=dist.ReduceOp.SUM,
                )

                tokens_seen += tokens_tensor.item()

            else:
                tokens_seen += local_tokens

            global_step += 1

            # ------------------------------------------------
            # Learning rate
            # ------------------------------------------------

            learning_rate_change(
                global_step=global_step,
                total_training_steps=total_steps,
                warmup_percent=0.20,
                optimizer=optimizer,
            )

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_dataloader,
                    val_dataloader,
                    device,
                    eval_iter,
                )

                if is_main_process():

                    train_losses.append(train_loss)

                    val_losses.append(val_loss)

                    track_tokens_seen.append(tokens_seen)

                    print(
                        f"Epoch {epoch + 1} "
                        f"(Step "
                        f"{global_step:06d}): "
                        f"Train loss "
                        f"{train_loss:.3f}, "
                        f"Val loss "
                        f"{val_loss:.3f}, "
                        f"LR "
                        f"{optimizer.param_groups[0]['lr']:.3e}"
                    )

            # ------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------

            if (
                create_checkpoints
                and global_step > 0
                and global_step % checkpoint_freq == 0
            ):

                if is_distributed():
                    dist.barrier()

                if is_main_process():

                    save_checkpoint(
                        latest_ckpt_path,
                        unwrap_model(model),
                        optimizer,
                        epoch,
                        global_step,
                        tokens_seen,
                        train_losses,
                        val_losses,
                        track_tokens_seen,
                    )

                if is_distributed():
                    dist.barrier()

        # ----------------------------------------------------
        # End-of-epoch sample
        # ----------------------------------------------------

        if is_main_process():

            generate_and_print_sample(
                unwrap_model(model),
                tokenizer,
                device,
                start_context,
            )

        # ----------------------------------------------------
        # End-of-epoch checkpoint
        # ----------------------------------------------------

        if create_checkpoints:

            if is_distributed():
                dist.barrier()

            if is_main_process():

                raw_model = unwrap_model(model)

                epoch_ckpt_path = os.path.join(
                    checkpoint_path,
                    f"epoch_{epoch + 1}.pt",
                )

                save_checkpoint(
                    epoch_ckpt_path,
                    raw_model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    tokens_seen,
                    train_losses,
                    val_losses,
                    track_tokens_seen,
                )

                save_checkpoint(
                    latest_ckpt_path,
                    raw_model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    tokens_seen,
                    train_losses,
                    val_losses,
                    track_tokens_seen,
                )

            if is_distributed():
                dist.barrier()

    return (
        train_losses,
        val_losses,
        track_tokens_seen,
    )


# ============================================================
# Evaluation
# ============================================================


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

    train_loss = evaluate_loader(
        model,
        train_dataloader,
        device,
        eval_iter,
    )
    val_loss = evaluate_loader(
        model,
        val_dataloader,
        device,
        eval_iter,
    )

    model.train()
    return train_loss, val_loss


def evaluate_loader(
    model,
    dataloader,
    device,
    eval_iter,
):

    total_loss = torch.zeros(
        1,
        device=device,
        dtype=torch.float64,
    )

    total_tokens = torch.zeros(
        1,
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():

        for batch_idx, (
            input_batch,
            target_batch,
        ) in enumerate(dataloader):

            if eval_iter is not None and batch_idx >= eval_iter:
                break

            loss = calc_batch_cost(
                input_batch,
                target_batch,
                model,
                device,
            )

            num_tokens = target_batch.numel()

            total_loss += loss.detach().double() * num_tokens

            total_tokens += num_tokens

    if is_distributed():

        dist.all_reduce(
            total_loss,
            op=dist.ReduceOp.SUM,
        )

        dist.all_reduce(
            total_tokens,
            op=dist.ReduceOp.SUM,
        )

    return (total_loss / total_tokens.clamp(min=1)).item()


# ============================================================
# Generation
# ============================================================


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
    initial_lr=1e-4,
    peak_lr=1e-1,
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
        # Linear increase (warmup)
        progress = global_step / max(warmup_steps, 1)
        lr = initial_lr + progress * lr_increment
    else:
        # Cosine decay
        decay_steps = total_training_steps - warmup_steps
        progress = (global_step - warmup_steps) / decay_steps

        progress = min(progress, 1.0)  # guard against overshoot

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (peak_lr - min_lr) * cosine_decay

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


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
