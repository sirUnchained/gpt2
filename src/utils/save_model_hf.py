from huggingface_hub import HfApi, create_repo, notebook_login

from configs.model_configs import GPT_configs


import os
from huggingface_hub import HfApi, create_repo, notebook_login
from configs.model_configs import GPT_configs


def save_model_hf(cfg: GPT_configs):
    """
    ## Upload a trained GPT model and its configuration to the Hugging Face Hub.

    Logs into Hugging Face (interactive), creates or retrieves a repository,
    and uploads the model weights (`pytorch_model.bin`) and configuration (`configs.json`)
    from the local folder specified in `cfg.save_model_path`.

    Args:
        cfg (GPT_configs): Configuration object containing:
            - save_model_path (str): Local directory where model files are stored.
            - repo_id (str): Hugging Face Hub repository ID (e.g., "username/model-name").

    Returns:
        None
    """

    folder_checkpoints_name = cfg.checkpoints_path
    folder_weights_name = cfg.save_model_path
    model_weights_name = "pytorch_model.bin"
    config_file_name = "configs.json"

    model_weights_path = os.path.join(folder_weights_name, model_weights_name)
    config_file_path = os.path.join(folder_weights_name, config_file_name)

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    repo_id = cfg.repo_id
    notebook_login()

    api = HfApi()

    print(f"Attempting to create/get repository: {repo_id}")
    create_repo(repo_id=repo_id, exist_ok=True, private=False)
    print(f"Repository {repo_id} created or already exists.")

    print(f"Uploading model weights ({model_weights_name})...")
    api.upload_file(
        path_or_fileobj=model_weights_path,
        path_in_repo=model_weights_name,
        repo_id=repo_id,
        commit_message="Add PyTorch model weights",
    )

    print(f"Uploading model configuration ({config_file_name})...")
    api.upload_file(
        path_or_fileobj=config_file_path,
        path_in_repo=config_file_name,
        repo_id=repo_id,
        commit_message="Add model configuration",
    )

    if cfg.create_checkpoints:
        print(f"Uploading model checkpoints ({folder_checkpoints_name})...")
        api.upload_folder(
            folder_path=folder_checkpoints_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model checkpoints",
        )

    print(
        f"Successfully pushed model to Hugging Face Hub: https://huggingface.co/{repo_id}"
    )
