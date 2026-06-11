import logging
import os
from typing import Any
from google import genai
from google.genai.errors import ClientError, ServerError
from .utils import get_gemini_api_keys, get_gemini_models
from .exceptions import AllClientsFailed
import ast
import re

try:
    from langsmith import wrappers, traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    wrappers = None
    traceable = None
    get_current_run_tree = None

if traceable is None:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _format_error(e: Exception) -> str:
    s = str(e)
    match = re.search(r"^(\d+)\s+([A-Z_]+)\.\s+(.*)$", s, re.DOTALL)

    if match:
        code, status, details_str = match.groups()
        try:
            details = ast.literal_eval(details_str)
            if isinstance(details, dict) and "error" in details:
                message = details["error"].get("message", "Unknown error")
                return f"{code} {status}. {message}"
        except:
            pass

    return s


def _should_think(contents: Any) -> bool:
    """
    Determines whether the model should perform reasoning/thinking.
    Thinking is only enabled if:
    1. The estimated token count of the prompt is > 100 tokens (~400 characters).
    2. OR there are any files (images, binary parts, blobs, etc.) sent in the prompt.
    """
    if contents is None:
        return False

    def is_file_like(item: Any) -> bool:
        if item is None:
            return False
        if isinstance(item, str):
            return False
        if isinstance(item, dict):
            if "text" in item and len(item) == 1:
                return False
            return True
        item_type_name = type(item).__name__
        if item_type_name in ("Image", "PILImage", "Part", "Blob") or "Image" in str(type(item)):
            if hasattr(item, "text") and item.text is not None:
                if not hasattr(item, "inline_data") and not hasattr(item, "file_data"):
                    return False
                if getattr(item, "inline_data", None) is None and getattr(item, "file_data", None) is None:
                    return False
            return True
        if isinstance(item, (bytes, bytearray)):
            return True
        return False

    total_char_count = 0
    has_files = False
    items = contents if isinstance(contents, list) else [contents]

    for item in items:
        if hasattr(item, "parts") and item.parts is not None:
            parts = item.parts if isinstance(item.parts, list) else [item.parts]
            for part in parts:
                if is_file_like(part):
                    has_files = True
                elif hasattr(part, "text") and part.text:
                    total_char_count += len(part.text)
                elif isinstance(part, str):
                    total_char_count += len(part)
        elif is_file_like(item):
            has_files = True
        elif isinstance(item, str):
            total_char_count += len(item)
        elif isinstance(item, dict):
            if "parts" in item:
                parts = item["parts"] if isinstance(item["parts"], list) else [item["parts"]]
                for part in parts:
                    if is_file_like(part):
                        has_files = True
                    elif isinstance(part, dict) and "text" in part:
                        total_char_count += len(str(part["text"]))
                    elif isinstance(part, str):
                        total_char_count += len(part)
            elif "text" in item:
                total_char_count += len(str(item["text"]))
            else:
                has_files = True

    # 1 token is ~4 characters, so 100 tokens is ~400 characters
    estimated_tokens = total_char_count / 4.0
    return has_files or (estimated_tokens > 100)


def _apply_default_thinking_config(model: str, config: Any, contents: Any) -> Any:
    """
    Applies minimal thinking/reasoning config by default to Gemini 3 and thinking models,
    unless reasoning is explicitly set by the user or the prompt length > 100 tokens or contains files.
    """
    if isinstance(config, dict):
        if "thinking_config" in config:
            return config
    elif hasattr(config, "thinking_config"):
        if getattr(config, "thinking_config", None) is not None:
            return config

    is_gemini_3 = "gemini-3" in model
    is_thinking_model = "thinking" in model

    if not is_gemini_3 and not is_thinking_model:
        return config

    # Enable full reasoning if prompt exceeds token threshold or has files
    if _should_think(contents):
        return config

    # Otherwise, minimize/disable reasoning by default
    default_thinking = {"thinking_budget": 0}

    if config is None:
        return {"thinking_config": default_thinking}
    elif isinstance(config, dict):
        new_config = dict(config)
        new_config["thinking_config"] = default_thinking
        return new_config
    else:
        try:
            from google.genai import types
            new_config = config.model_copy(update={
                "thinking_config": types.ThinkingConfig(**default_thinking)
            })
            return new_config
        except Exception:
            try:
                config.thinking_config = default_thinking
            except Exception:
                pass
            return config


def _calculate_gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Computes pricing per million tokens based on model name.
    """
    model_lower = model.lower()

    # Gemma 4 is free on Google AI Studio
    if "gemma-4" in model_lower:
        input_rate = 0.0
        output_rate = 0.0

    # Gemini 3.5 Flash
    elif "gemini-3.5-flash" in model_lower:
        input_rate = 1.50 / 1_000_000
        output_rate = 9.00 / 1_000_000

    # Gemini 3.5 Live Translate
    elif "gemini-3.5-live-translate" in model_lower:
        input_rate = 3.50 / 1_000_000
        output_rate = 21.00 / 1_000_000

    # Gemini 3.1 Pro Preview
    elif "gemini-3.1-pro" in model_lower:
        if input_tokens <= 200_000:
            input_rate = 2.00 / 1_000_000
            output_rate = 12.00 / 1_000_000
        else:
            input_rate = 4.00 / 1_000_000
            output_rate = 18.00 / 1_000_000

    # Gemini 3.1 Flash-Lite
    elif "gemini-3.1-flash-lite" in model_lower:
        input_rate = 0.25 / 1_000_000
        output_rate = 1.50 / 1_000_000

    # Gemini 3.1 Flash Live Preview
    elif "gemini-3.1-flash-live" in model_lower:
        input_rate = 0.75 / 1_000_000
        output_rate = 4.50 / 1_000_000

    # Gemini 3.1 Flash Image
    elif "gemini-3.1-flash-image" in model_lower:
        input_rate = 0.50 / 1_000_000
        output_rate = 3.00 / 1_000_000

    # Gemini 3.1 Flash TTS Preview
    elif "gemini-3.1-flash-tts" in model_lower:
        input_rate = 1.00 / 1_000_000
        output_rate = 20.00 / 1_000_000

    # Gemini 3 Flash Preview
    elif "gemini-3-flash" in model_lower:
        input_rate = 0.50 / 1_000_000
        output_rate = 3.00 / 1_000_000

    # Gemini 3 Pro Image
    elif "gemini-3-pro-image" in model_lower:
        input_rate = 2.00 / 1_000_000
        output_rate = 12.00 / 1_000_000

    # Gemini 2.5 Pro
    elif "gemini-2.5-pro" in model_lower:
        if input_tokens <= 200_000:
            input_rate = 1.25 / 1_000_000
            output_rate = 10.00 / 1_000_000
        else:
            input_rate = 2.50 / 1_000_000
            output_rate = 15.00 / 1_000_000

    # Gemini 2.5 Computer Use Preview
    elif "gemini-2.5-computer-use" in model_lower:
        if input_tokens <= 200_000:
            input_rate = 1.25 / 1_000_000
            output_rate = 10.00 / 1_000_000
        else:
            input_rate = 2.50 / 1_000_000
            output_rate = 15.00 / 1_000_000

    # Gemini 2.5 Flash Native Audio
    elif "gemini-2.5-flash-native-audio" in model_lower:
        input_rate = 0.50 / 1_000_000
        output_rate = 2.00 / 1_000_000

    # Gemini 2.5 Flash Image
    elif "gemini-2.5-flash-image" in model_lower:
        input_rate = 0.30 / 1_000_000
        output_rate = 30.00 / 1_000_000

    # Gemini 2.5 Flash Preview TTS
    elif "gemini-2.5-flash-preview-tts" in model_lower:
        input_rate = 0.50 / 1_000_000
        output_rate = 10.00 / 1_000_000

    # Gemini 2.5 Pro Preview TTS
    elif "gemini-2.5-pro-preview-tts" in model_lower:
        input_rate = 1.00 / 1_000_000
        output_rate = 20.00 / 1_000_000

    # Gemini 2.5 Flash
    elif "gemini-2.5-flash" in model_lower:
        input_rate = 0.30 / 1_000_000
        output_rate = 2.50 / 1_000_000

    # Gemini 2.5 Flash-Lite
    elif "gemini-2.5-flash-lite" in model_lower:
        input_rate = 0.10 / 1_000_000
        output_rate = 0.40 / 1_000_000

    # Gemini Robotics Embodied Reasoning
    elif "gemini-robotics-er" in model_lower:
        input_rate = 1.00 / 1_000_000
        output_rate = 5.00 / 1_000_000

    # Gemini 2.0 Flash-Lite (deprecated/shut down)
    elif "gemini-2.0-flash-lite" in model_lower:
        input_rate = 0.075 / 1_000_000
        output_rate = 0.30 / 1_000_000

    # Gemini 2.0 Flash (deprecated/shut down)
    elif "gemini-2.0-flash" in model_lower:
        input_rate = 0.10 / 1_000_000
        output_rate = 0.40 / 1_000_000

    # Legacy / general / fallback pricing
    else:
        input_rate = 0.075 / 1_000_000
        output_rate = 0.30 / 1_000_000
        if "lite" in model_lower:
            input_rate = 0.0375 / 1_000_000
            output_rate = 0.15 / 1_000_000
        elif "pro" in model_lower:
            input_rate = 1.25 / 1_000_000
            output_rate = 5.00 / 1_000_000
        elif "gemma" in model_lower:
            input_rate = 0.0375 / 1_000_000
            output_rate = 0.15 / 1_000_000

    return (input_tokens * input_rate) + (output_tokens * output_rate)


def _inject_cost_serialization(response: Any, cost: float) -> None:
    if hasattr(response, "model_dump"):
        try:
            orig_model_dump = response.model_dump
            def custom_model_dump(*args, **kwargs):
                d = orig_model_dump(*args, **kwargs)
                if "usage_metadata" in d and d["usage_metadata"] is not None:
                    d["usage_metadata"]["total_cost"] = cost
                return d
            object.__setattr__(response, "model_dump", custom_model_dump)
        except Exception:
            pass

    if hasattr(response, "dict"):
        try:
            orig_dict = response.dict
            def custom_dict(*args, **kwargs):
                d = orig_dict(*args, **kwargs)
                if "usage_metadata" in d and d["usage_metadata"] is not None:
                    d["usage_metadata"]["total_cost"] = cost
                return d
            object.__setattr__(response, "dict", custom_dict)
        except Exception:
            pass


class GeminiRotationClient:
    def __init__(self, tracing_extra: Any = None):
        self.api_keys = get_gemini_api_keys()
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in environment variables.")

        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self.models = get_gemini_models()

        tracing_enabled = (
            os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
            or os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        )
        if wrappers is not None and tracing_enabled:
            default_tracing_extra = {
                "tags": ["gemini-rotate", "gemini", "python"],
                "metadata": {
                    "integration": "gemini-rotate",
                },
            }
            if tracing_extra:
                tags = list(set(default_tracing_extra["tags"] + tracing_extra.get("tags", [])))
                metadata = {**default_tracing_extra["metadata"], **tracing_extra.get("metadata", {})}
                merged_extra = {**default_tracing_extra, **tracing_extra, "tags": tags, "metadata": metadata}
            else:
                merged_extra = default_tracing_extra

            wrapped_clients = []
            for client_idx, client in enumerate(self.clients):
                client_id = f"Client-{client_idx + 1}"
                client_extra = {
                    **merged_extra,
                    "metadata": {
                        **merged_extra.get("metadata", {}),
                        "client_id": client_id
                    }
                }
                wrapped_clients.append(wrappers.wrap_gemini(client, tracing_extra=client_extra))
            self.clients = wrapped_clients
            logger.info("Wrapped Google GenAI clients with LangSmith tracing.")


    @traceable(name="GeminiRotationClient.generate_content", run_type="llm")
    async def generate_content(
        self,
        contents: Any,
        config: Any = None,
    ):
        """
        Generates content using rotated clients and model pairs.
        Iterates through models in pairs (Primary, Secondary).
        For each pair, iterates through all available API keys.
        """
        model_groups = [self.models[i : i + 2] for i in range(0, len(self.models), 2)]

        total_groups = len(model_groups)

        for group_idx, group in enumerate(model_groups):
            primary_model = group[0]
            secondary_model = group[1] if len(group) > 1 else None

            logger.info(
                f"Processing Model Group {group_idx + 1}/{total_groups}: Primary='{primary_model}', Secondary='{secondary_model}'"
            )

            for client_idx, client in enumerate(self.clients):
                client_id = f"Client-{client_idx + 1}"

                try:
                    logger.debug(f"[{client_id}] Attempting Primary: {primary_model}")
                    actual_config = _apply_default_thinking_config(primary_model, config, contents)
                    call_kwargs = {}
                    if actual_config is not None:
                        call_kwargs["config"] = actual_config
                    response = await client.aio.models.generate_content(
                        model=primary_model, contents=contents, **call_kwargs
                    )
                    logger.info(f"[{client_id}] Primary ({primary_model}) succeeded")
                    if get_current_run_tree is not None:
                        run_tree = get_current_run_tree()
                        if run_tree:
                            outputs = {"output": response.text}
                            usage = getattr(response, "usage_metadata", None)
                            if usage:
                                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                                total_tokens = getattr(usage, "total_token_count", 0) or 0
                                cost = _calculate_gemini_cost(primary_model, input_tokens, output_tokens)
                                outputs["usage_metadata"] = {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": total_tokens,
                                    "total_cost": cost,
                                }
                                _inject_cost_serialization(response, cost)
                                run_tree.metadata.update({
                                    "usage_metadata": {
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "total_tokens": total_tokens,
                                        "total_cost": cost,
                                    }
                                })
                            run_tree.outputs = outputs
                            run_tree.metadata.update({
                                "succeeded_client": client_id,
                                "succeeded_model": primary_model,
                                "ls_model_name": primary_model,
                                "ls_provider": "google",
                            })
                            try:
                                run_tree.patch()
                            except Exception:
                                pass
                    try:
                        response.__dict__["client_id"] = client_id
                        response.__dict__["model"] = primary_model
                    except Exception:
                        pass
                    try:
                        response.client_id = client_id
                        response.model = primary_model
                    except Exception:
                        pass
                    return response

                except (ClientError, ServerError) as e:
                    logger.warning(
                        f"[{client_id}] Primary ({primary_model}) failed: {_format_error(e)}"
                    )

                    if secondary_model:
                        try:
                            logger.debug(
                                f"[{client_id}] Attempting Secondary: {secondary_model}"
                            )
                            actual_config = _apply_default_thinking_config(secondary_model, config, contents)
                            call_kwargs = {}
                            if actual_config is not None:
                                call_kwargs["config"] = actual_config
                            response = await client.aio.models.generate_content(
                                model=secondary_model, contents=contents, **call_kwargs
                            )
                            logger.info(
                                f"[{client_id}] Secondary ({secondary_model}) succeeded"
                            )
                            if get_current_run_tree is not None:
                                run_tree = get_current_run_tree()
                                if run_tree:
                                    outputs = {"output": response.text}
                                    usage = getattr(response, "usage_metadata", None)
                                    if usage:
                                        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                                        total_tokens = getattr(usage, "total_token_count", 0) or 0
                                        cost = _calculate_gemini_cost(secondary_model, input_tokens, output_tokens)
                                        outputs["usage_metadata"] = {
                                            "input_tokens": input_tokens,
                                            "output_tokens": output_tokens,
                                            "total_tokens": total_tokens,
                                            "total_cost": cost,
                                        }
                                        _inject_cost_serialization(response, cost)
                                        run_tree.metadata.update({
                                            "usage_metadata": {
                                                "input_tokens": input_tokens,
                                                "output_tokens": output_tokens,
                                                "total_tokens": total_tokens,
                                                "total_cost": cost,
                                            }
                                        })
                                    run_tree.outputs = outputs
                                    run_tree.metadata.update({
                                        "succeeded_client": client_id,
                                        "succeeded_model": secondary_model,
                                        "ls_model_name": secondary_model,
                                        "ls_provider": "google",
                                    })
                                    try:
                                        run_tree.patch()
                                    except Exception:
                                        pass
                            try:
                                response.__dict__["client_id"] = client_id
                                response.__dict__["model"] = secondary_model
                            except Exception:
                                pass
                            try:
                                response.client_id = client_id
                                response.model = secondary_model
                            except Exception:
                                pass
                            return response
                        except (ClientError, ServerError) as e2:
                            logger.warning(
                                f"[{client_id}] Secondary ({secondary_model}) failed: {_format_error(e2)}"
                            )

        raise AllClientsFailed(
            f"All {len(self.clients)} agents failed across all {len(self.models)} models."
        )


    @traceable(name="GeminiRotationClient.generate_content_sync", run_type="llm")
    def generate_content_sync(
        self,
        contents: Any,
        config: Any = None,
    ):
        """
        Generates content synchronously using rotated clients and model pairs.
        Iterates through models in pairs (Primary, Secondary).
        For each pair, iterates through all available API keys.
        """
        model_groups = [self.models[i : i + 2] for i in range(0, len(self.models), 2)]

        total_groups = len(model_groups)

        for group_idx, group in enumerate(model_groups):
            primary_model = group[0]
            secondary_model = group[1] if len(group) > 1 else None

            logger.info(
                f"Processing Model Group {group_idx + 1}/{total_groups}: Primary='{primary_model}', Secondary='{secondary_model}'"
            )

            for client_idx, client in enumerate(self.clients):
                client_id = f"Client-{client_idx + 1}"

                try:
                    logger.debug(f"[{client_id}] Attempting Primary: {primary_model}")
                    actual_config = _apply_default_thinking_config(primary_model, config, contents)
                    call_kwargs = {}
                    if actual_config is not None:
                        call_kwargs["config"] = actual_config
                    response = client.models.generate_content(
                        model=primary_model, contents=contents, **call_kwargs
                    )
                    logger.info(f"[{client_id}] Primary ({primary_model}) succeeded")
                    if get_current_run_tree is not None:
                        run_tree = get_current_run_tree()
                        if run_tree:
                            outputs = {"output": response.text}
                            usage = getattr(response, "usage_metadata", None)
                            if usage:
                                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                                total_tokens = getattr(usage, "total_token_count", 0) or 0
                                cost = _calculate_gemini_cost(primary_model, input_tokens, output_tokens)
                                outputs["usage_metadata"] = {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": total_tokens,
                                    "total_cost": cost,
                                }
                                _inject_cost_serialization(response, cost)
                                run_tree.metadata.update({
                                    "usage_metadata": {
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "total_tokens": total_tokens,
                                        "total_cost": cost,
                                    }
                                })
                            run_tree.outputs = outputs
                            run_tree.metadata.update({
                                "succeeded_client": client_id,
                                "succeeded_model": primary_model,
                                "ls_model_name": primary_model,
                                "ls_provider": "google",
                            })
                            try:
                                run_tree.patch()
                            except Exception:
                                pass
                    try:
                        response.__dict__["client_id"] = client_id
                        response.__dict__["model"] = primary_model
                    except Exception:
                        pass
                    try:
                        response.client_id = client_id
                        response.model = primary_model
                    except Exception:
                        pass
                    return response

                except (ClientError, ServerError) as e:
                    logger.warning(
                        f"[{client_id}] Primary ({primary_model}) failed: {_format_error(e)}"
                    )

                    if secondary_model:
                        try:
                            logger.debug(
                                f"[{client_id}] Attempting Secondary: {secondary_model}"
                            )
                            actual_config = _apply_default_thinking_config(secondary_model, config, contents)
                            call_kwargs = {}
                            if actual_config is not None:
                                call_kwargs["config"] = actual_config
                            response = client.models.generate_content(
                                model=secondary_model, contents=contents, **call_kwargs
                            )
                            logger.info(
                                f"[{client_id}] Secondary ({secondary_model}) succeeded"
                            )
                            if get_current_run_tree is not None:
                                run_tree = get_current_run_tree()
                                if run_tree:
                                    outputs = {"output": response.text}
                                    usage = getattr(response, "usage_metadata", None)
                                    if usage:
                                        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                                        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                                        total_tokens = getattr(usage, "total_token_count", 0) or 0
                                        cost = _calculate_gemini_cost(secondary_model, input_tokens, output_tokens)
                                        outputs["usage_metadata"] = {
                                            "input_tokens": input_tokens,
                                            "output_tokens": output_tokens,
                                            "total_tokens": total_tokens,
                                            "total_cost": cost,
                                        }
                                        _inject_cost_serialization(response, cost)
                                        run_tree.metadata.update({
                                            "usage_metadata": {
                                                "input_tokens": input_tokens,
                                                "output_tokens": output_tokens,
                                                "total_tokens": total_tokens,
                                                "total_cost": cost,
                                            }
                                        })
                                    run_tree.outputs = outputs
                                    run_tree.metadata.update({
                                        "succeeded_client": client_id,
                                        "succeeded_model": secondary_model,
                                        "ls_model_name": secondary_model,
                                        "ls_provider": "google",
                                    })
                                    try:
                                        run_tree.patch()
                                    except Exception:
                                        pass
                            try:
                                response.__dict__["client_id"] = client_id
                                response.__dict__["model"] = secondary_model
                            except Exception:
                                pass
                            try:
                                response.client_id = client_id
                                response.model = secondary_model
                            except Exception:
                                pass
                            return response
                            
                        except (ClientError, ServerError) as e2:
                            logger.warning(
                                f"[{client_id}] Secondary ({secondary_model}) failed: {_format_error(e2)}"
                            )

        raise AllClientsFailed(
            f"All {len(self.clients)} agents failed across all {len(self.models)} models."
        )
