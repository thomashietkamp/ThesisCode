# --- START OF FILE utils.py ---
import torch
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_DEVICE = None


def get_device() -> torch.device:
    """
    Detects and returns the appropriate torch device (CUDA, MPS, or CPU).
    Caches the result for subsequent calls.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    _DEVICE = device
    return device


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """
    Determines the appropriate torch dtype based on the device.
    Prefers bfloat16 for CUDA Ampere+, float16 for others, float32 as fallback.
    """
    if device.type == 'cuda':
        # Check for Ampere architecture or higher for bfloat16 support
        compute_capability = torch.cuda.get_device_capability(device)
        if compute_capability[0] >= 8:
            logger.info("CUDA device supports bfloat16. Using torch.bfloat16.")
            return torch.bfloat16
        else:
            logger.info("CUDA device detected. Using torch.float16.")
            return torch.float16
    elif device.type == 'mps':
        # MPS support for float16 can be tricky, float32 is safer sometimes
        # Let's default to float16 but be aware it might need adjustment
        logger.info("MPS device detected. Using torch.float16.")
        return torch.float16
    else:
        logger.info("CPU device detected. Using torch.float32.")
        return torch.float32


def check_huggingface_auth(token: Optional[str] = None) -> bool:
    """
    Checks if the user is authenticated with Hugging Face Hub.
    Optionally uses a provided token.

    Args:
        token (Optional[str]): A Hugging Face API token to use for authentication.

    Returns:
        bool: True if authenticated, False otherwise.

    Raises:
        SystemExit: If authentication fails or is required but not configured.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Please install it: pip install huggingface_hub")
        raise SystemExit(1)

    api = HfApi()
    user_info = None
    try:
        if token:
            user_info = api.whoami(token=token)
            logger.info(
                f"Authenticated using provided token for user: {user_info.get('name')}")
        else:
            user_info = api.whoami()  # Checks cached token
            logger.info(
                f"Authenticated using cached token for user: {user_info.get('name')}")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            logger.error("Hugging Face authentication required.")
            logger.error(
                "Please login using 'huggingface-cli login' or provide a valid HF token.")
            if token:
                logger.error("The provided token might be invalid or expired.")
            else:
                logger.error("No cached token found or the token is invalid.")
        else:
            logger.error(f"Error checking Hugging Face authentication: {e}")
        raise SystemExit(1)  # Exit if not authenticated
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during Hugging Face authentication check: {e}")
        raise SystemExit(1)


# --- END OF FILE utils.py ---
