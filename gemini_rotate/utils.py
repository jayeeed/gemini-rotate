import os
import json
import logging

logger = logging.getLogger(__name__)

def get_gemini_api_keys() -> list[str]:
    """
    Retrieves all environment variables starting with GEMINI_API_KEY_
    and returns a list of their values, preserving order.
    Also checks for GEMINI_API_KEY (without suffix).
    """
    keys = []
    
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key:
        keys.append(single_key)

    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            break
            
    # Preserve order while removing duplicates and empty strings
    unique_keys = list(dict.fromkeys([k for k in keys if k]))
    
    if not unique_keys:
        logger.warning("No GEMINI_API_KEY found in environment variables.")
        
    return unique_keys

def get_gemini_models() -> list[str]:
    """
    Retrieves the list of Gemini models from the environment variable GEMINI_MODELS.
    If not found, returns the user-defined default list.
    """
    default_models = [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it"
    ]
    
    env_models_str = os.getenv("GEMINI_MODELS")
    if env_models_str:
        try:
            models = json.loads(env_models_str)
            if isinstance(models, list):
                return models
            else:
                logger.warning("GEMINI_MODELS is not a JSON list. Using default models.")
        except json.JSONDecodeError:
            logger.warning("Failed to parse GEMINI_MODELS. Using default models.")
            
    return default_models
