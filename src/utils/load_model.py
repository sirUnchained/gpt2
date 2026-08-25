import os
import torch
from configs.model_configs import GPT_configs


def load_model_if_exists(cfg: GPT_configs, model: torch.nn.Module, device="cpu"):
    """
    ## Load model weights from a local file if it exists.

    This function checks for a saved PyTorch model weights file (`pytorch_model.bin`)
    inside the directory specified by `cfg.save_model_path`. If the file exists,
    it loads the state dictionary into the provided model and returns `True`.
    Otherwise, it prints a message and returns `False`.

    Args:
        cfg (GPT_configs): Configuration object containing:
            - save_model_path (str): Directory where the model weights file is stored.
        model (torch.nn.Module): The model instance into which the weights will be loaded.
        device (str, optional): Device to map the loaded state dict to (e.g., "cpu", "cuda").
            Default is "cpu".

    Returns:
        bool: `True` if weights were successfully loaded, `False` otherwise.
    """
    folder_name = cfg.save_model_path
    model_weights_name = "pytorch_model.bin"

    model_weights_path = os.path.join(folder_name, model_weights_name)

    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_weights_path}")
        return True
    else:
        print(f"No model weights found at {model_weights_path}")
        return False
