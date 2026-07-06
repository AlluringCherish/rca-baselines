import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def _read_json_config(path: str | None) -> Dict[str, Any]:
    if path:
        config_path = Path(path)
    else:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists() and not path:
        return {}
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"API config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
    for key in keys:
        lower = key.lower()
        if lower in config and config[lower] not in (None, ""):
            return config[lower]
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


def _config_file_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        lower = key.lower()
        if lower in config and config[lower] not in (None, ""):
            return config[lower]
        if key in config and config[key] not in (None, ""):
            return config[key]
    for key in keys:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
    return default


def _load_config() -> Dict[str, Any]:
    raw_config = _read_json_config(os.environ.get("API_CONFIG_PATH"))
    provider_config = raw_config.get("openrouter", raw_config)

    api_key = _config_value(
        provider_config,
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
        "api_key",
    )
    model = _config_file_value(
        provider_config,
        "OPENROUTER_MODEL",
        "OPENAI_MODEL",
        "MODEL",
        "model",
    )
    base_url = _config_file_value(
        provider_config,
        "OPENROUTER_BASE_URL",
        "OPENAI_BASE_URL",
        "API_BASE",
        "base_url",
        default=DEFAULT_OPENROUTER_BASE_URL,
    )
    temperature = float(_config_file_value(
        provider_config,
        "OPENROUTER_TEMPERATURE",
        "OPENAI_TEMPERATURE",
        "TEMPERATURE",
        "temperature",
        default=0.0,
    ))
    reasoning_effort = _config_file_value(
        provider_config,
        "OPENROUTER_REASONING_EFFORT",
        "OPENAI_REASONING_EFFORT",
        "REASONING_EFFORT",
        "reasoning_effort",
    )
    seed = _config_file_value(
        provider_config,
        "OPENROUTER_SEED",
        "OPENAI_SEED",
        "SEED",
        "seed",
    )
    if seed is not None:
        seed = int(seed)

    default_headers = {}
    referer = _config_value(provider_config, "OPENROUTER_HTTP_REFERER", "HTTP_REFERER", "http_referer")
    title = _config_value(provider_config, "OPENROUTER_APP_TITLE", "APP_TITLE", "app_title")
    if referer:
        default_headers["HTTP-Referer"] = referer
    if title:
        default_headers["X-Title"] = title

    return {
        "MODEL": model,
        "API_KEY": api_key,
        "API_BASE": base_url,
        "TEMPERATURE": temperature,
        "REASONING_EFFORT": reasoning_effort,
        "SEED": seed,
        "DEFAULT_HEADERS": default_headers,
    }


configs = _load_config()

_TOKEN_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
}


def reset_token_usage() -> None:
    _TOKEN_USAGE["input_tokens"] = 0
    _TOKEN_USAGE["output_tokens"] = 0


def get_token_usage() -> Dict[str, int]:
    return dict(_TOKEN_USAGE)


def _usage_value(usage: Any, *keys: str) -> int:
    if usage is None:
        return 0
    for key in keys:
        if isinstance(usage, dict) and usage.get(key) is not None:
            return int(usage[key])
        if hasattr(usage, key) and getattr(usage, key) is not None:
            return int(getattr(usage, key))
    return 0


def _record_token_usage(usage: Any) -> None:
    _TOKEN_USAGE["input_tokens"] += _usage_value(usage, "prompt_tokens", "input_tokens")
    _TOKEN_USAGE["output_tokens"] += _usage_value(usage, "completion_tokens", "output_tokens")


def require_api_config() -> None:
    missing = []
    if not configs.get("API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if not configs.get("MODEL"):
        missing.append("OPENROUTER_MODEL")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing API configuration: {names}. "
            "Set OpenRouter environment variables or provide API_CONFIG_PATH."
        )


def get_chat_completion(messages, temperature: Optional[float] = None, tools=None, parallel_tool_calls: bool = False):
    from openai import OpenAI

    require_api_config()
    if temperature is None:
        temperature = configs["TEMPERATURE"]

    client_args = {"api_key": configs["API_KEY"]}
    if configs["API_BASE"]:
        client_args["base_url"] = configs["API_BASE"]
    if configs["DEFAULT_HEADERS"]:
        client_args["default_headers"] = configs["DEFAULT_HEADERS"]
    client = OpenAI(**client_args)

    request_args: Dict[str, Any] = {
        "model": configs["MODEL"],
        "messages": messages,
        "temperature": temperature,
    }
    if configs.get("REASONING_EFFORT"):
        request_args["reasoning_effort"] = configs["REASONING_EFFORT"]
    if configs.get("SEED") is not None:
        request_args["seed"] = configs["SEED"]
    if tools:
        request_args["tools"] = tools
        request_args["parallel_tool_calls"] = parallel_tool_calls
    if "openrouter" in str(configs.get("API_BASE", "")).lower():
        request_args["extra_body"] = {"usage": {"include": True}}

    for _ in range(3):
        try:
            response = client.chat.completions.create(**request_args)
            _record_token_usage(getattr(response, "usage", None))
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            if "429" in str(e):
                print("Rate limit exceeded. Waiting for 1 second.")
                time.sleep(1)
                continue
            raise


if __name__ == "__main__":
    print(f"Model: {configs.get('MODEL')}")
    response = get_chat_completion([{"role": "user", "content": "123+321=?"}])
    print(response)
