from pprint import pprint
from pathlib import Path
import json
import sys
import os

import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchinfo import summary

from scripts.evaluate import (
    generate_text_with_temperature_topk,
    text_to_token_ids,
    token_ids_to_text,
)
from configs.model_configs import get_gpt_configs, GPT_configs
from src.data.dataset import create_dataloader, load_text_data
from src.training.train import train_model
from src.models.gpt_model import GPT_model
from src.utils.save_model_hf import save_model_hf, ensure_huggingface_login
from src.utils.load_model import load_model_if_exists


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_device():
    """
    Initialize distributed training if launched with torchrun.
    Returns:
        device, local_rank, distributed
    """

    distributed = is_distributed()

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend="nccl")

        device = torch.device(f"cuda:{local_rank}")

    else:
        local_rank = 0

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return device, local_rank, distributed


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():

    args = sys.argv[1:]

    device, local_rank, distributed = setup_device()

    if is_main_process():
        print("Welcome to the GPT model pipeline! " "loading stuff please wait ...")

        if distributed:
            print(
                f"Distributed training enabled. "
                f"World size: {torch.distributed.get_world_size()}"
            )

        print(f"Current device: {device}")

    cfg = get_gpt_configs()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    if cfg.repo_id != "INVALID":
        ensure_huggingface_login(cfg=cfg)

    if "-t" in args:

        # IMPORTANT:
        # Every rank creates its own model replica.
        model = GPT_model(cfg).to(device=device)

        if is_main_process():
            print("Model created.")

        try:
            train(
                model,
                cfg,
                device,
                distributed,
            )
        finally:
            cleanup_distributed()

        return 0

    # ---------------------------------------------------------
    # Generation / inspection
    # ---------------------------------------------------------

    # These modes should normally be launched without torchrun.

    model = GPT_model(cfg).to(device=device)

    if "-g" in args:

        if "-p" in args:
            prompt = args[args.index("-p") + 1]
        else:
            prompt = input("Enter your prompt: ")

        if load_model_if_exists(cfg, model, device):
            generate(model, cfg, prompt, 42)
        else:
            print("Model not found, try to train it first.")

        return 0

    if "-i" in args:

        print("Models configuration:")
        pprint(cfg.__dict__)

        print("Models architecture:")
        summary(model)

        return 0

    # ---------------------------------------------------------
    # Interactive mode
    # ---------------------------------------------------------

    while True:

        print(
            "Choose what to do: "
            "(q = quit, t = train, g = generate, i = model/config info)"
        )

        inp = input().lower()

        if inp == "q":
            print("Bye!")
            return 0

        elif inp == "t":

            try:
                train(
                    model,
                    cfg,
                    device,
                    distributed=False,
                )
            finally:
                cleanup_distributed()

            return 0

        elif inp == "g":

            if load_model_if_exists(cfg, model, device):

                prompt = input("Enter your prompt: ")

                generate(
                    model,
                    cfg,
                    prompt,
                    42,
                )

                return 0

            print("Model not found, try to train it first.")

        elif inp == "i":

            print("Models configuration:")
            pprint(cfg.__dict__)

            print("Models architecture:")
            summary(model)

        else:
            print("Unknown input, try again.")


<<<<<<< HEAD
def generate(
    model,
    cfg: GPT_configs,
    prompt: str,
    seed=None,
):

    if seed is not None:
        torch.manual_seed(seed)

=======
def generate(model, cfg: GPT_configs, prompt: str, seed=None):
>>>>>>> before-turning-code-into-distributed-gpus
    model.eval()

    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)

    generated_ids = generate_text_with_temperature_topk(
        model=model,
        idx=text_to_token_ids(
            text=prompt,
            tokenizer=tokenizer,
        ),
        context_size=cfg.context_length,
        max_new_tokens=25,
        top_k=90,
        temperature=0.7,
    )

    print(
        token_ids_to_text(
            generated_ids,
            tokenizer,
        )
    )


def train(
    model: GPT_model,
    cfg: GPT_configs,
    device,
    distributed: bool,
):

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    seed = 42

    # Different rank seeds are useful for dataloader shuffling,
    # while model initialization remains synchronized by DDP.
    torch.manual_seed(seed)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if is_main_process():
        print("Loading dataset...")

    text = load_text_data(cfg)

    train_ratio = 0.80

    split_idx = int(train_ratio * len(text))

    train_data = text[:split_idx]
    val_data = text[split_idx:]

    tokenizer = tiktoken.get_encoding(cfg.ticktoken_tokenizer)

    # ---------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------

    train_loader = create_dataloader(
        train_data,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=not distributed,
        num_workers=0,
        tokenizer_name=cfg.ticktoken_tokenizer,
        distributed=distributed,
        is_train=True,
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=cfg.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        tokenizer_name=cfg.ticktoken_tokenizer,
        distributed=distributed,
        is_train=False,
    )

<<<<<<< HEAD
    # ---------------------------------------------------------
    # Model / optimizer
    # ---------------------------------------------------------

    model.to(
        device=device,
        dtype=torch.bfloat16,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
=======
    # ==== SETUP EPOCHS DEVICE AND OPTIMIZER ====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=0.1)
>>>>>>> before-turning-code-into-distributed-gpus

    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=cfg.epochs,
        optimizer=optimizer,
        device=device,
        eval_freq=5,
        eval_iter=cfg.batch_size,
        start_context="Hello I am",
        tokenizer=tokenizer,
        create_checkpoints=cfg.create_checkpoints,
        use_checkpoints=cfg.use_checkpoints,
        checkpoint_freq=cfg.checkpoint_freq,
        checkpoint_path=cfg.checkpoints_path,
        distributed=distributed,
    )

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------

    if is_main_process():

        folder = Path(cfg.save_model_path)

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            folder / "pytorch_model.bin",
        )

        with open(
            folder / "configs.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                cfg.__dict__,
                f,
                indent=4,
            )

        save_model_hf(cfg)

        plot_losses(
            cfg.epochs,
            tokens_seen,
            train_losses,
            val_losses,
        )

        print("Training process finished.")


def plot_losses(
    epochs_seen,
    tokens_seen,
    train_losses,
    val_losses,
):

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(
        epochs_seen,
        train_losses,
        label="Training loss",
    )

    ax1.plot(
        epochs_seen,
        val_losses,
        linestyle="-.",
        label="Validation loss",
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax1.legend(loc="upper right")

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()

    ax2.plot(
        tokens_seen,
        train_losses,
        alpha=0,
    )

    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()

    fig.savefig("train-val-loss.png")

    plt.close(fig)


if __name__ == "__main__":
    main()
