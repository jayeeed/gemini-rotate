# Gemini Rotate

![Async](https://img.shields.io/badge/async-supported-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)

A lightweight Python library for Google Gemini API key rotation, valid model selection, and automatic fallback to "Lite" models on server errors. Supports both **Async** and **Sync** execution.

## 🚀 Features

- **✅ Automatic Key Rotation**: Seamlessly rotates through a list of API keys when quota is exhausted (`429`), permission denied (`403`), or any other API error occurs.
- **🔄 Smart Model Fallback**: Automatically downgrades specific models if server errors (`5xx`) persist.
- **⚡ Async & Sync Support**: Built on top of the `google-genai` client, offering both `async` (`generate_content`) and `sync` (`generate_content_sync`) methods for high-performance and standard applications.
- **🛡️ Robust Error Handling**: Implements exponential backoff before rotating keys or switching models.
- **📝 Concise Logging**: Logs only essential success/failure information (e.g., `400 INVALID_ARGUMENT`) to keep your console clean.
- **📊 Integrated LangSmith Tracing**: Zero-setup wrapper integration with LangSmith. Automatically traces requests, attributes success to specific API clients and models, and logs accurate pricing and token metadata.

## 📦 Installation
```bash
pip install gemini-rotate
```

## ⚡ Quick Start

1.  **Configure Environment**: Copy [.env.example](file:///Users/jayed/gits/gemini-rotate/.env.example) to `.env` and configure your API keys:
    ```env
    GEMINI_API_KEY_1="AIzaSy..."
    GEMINI_API_KEY_2="AI3yhj..."
    GEMINI_API_KEY_3="AIdf56..."
    ```

2.  **Run Code**:
    ```python
    import asyncio
    from gemini_rotate import GeminiRotationClient
    
    async def main():
        client = GeminiRotationClient()
        response = await client.generate_content("Hello, Gemini!")
        print(response.text)
    
    asyncio.run(main())
    ```

## 📖 Usage Guide

### Initialization
The client automatically loads API keys from your environment variables (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc.).

```python
client = GeminiRotationClient()
```

### Generating Content
The library provides both asynchronous (`generate_content`) and synchronous (`generate_content_sync`) methods. Both methods wrap the standard `google-genai` calls but add rotation and fallback logic.

#### 1. Async Text Generation
```python
import asyncio
from gemini_rotate import GeminiRotationClient
from dotenv import load_dotenv

load_dotenv()

async def generate_text():
    client = GeminiRotationClient()
    try:
        response = await client.generate_content(
            contents="Explain quantum computing in 50 words."
        )
        print(f"Generated text: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_text())
```

#### 2. Sync Text Generation
```python
from gemini_rotate import GeminiRotationClient
from dotenv import load_dotenv

load_dotenv()

def generate_text_sync():
    client = GeminiRotationClient()
    try:
        response = client.generate_content_sync(
            contents="Explain quantum computing in 50 words."
        )
        print(f"Generated text: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_text_sync()
```

#### 3. Advanced: Tool Calling & Structured Output (Async Example)
You can pass `tools` and `response_schema` (or `response_mime_type`) via the `config` parameter.

```python
import asyncio
from google import genai
from gemini_rotate import GeminiRotationClient
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Define a schema for structured output
class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    instructions: list[str]

async def generate_recipe():
    client = GeminiRotationClient()

    try:
        response = await client.generate_content(
            contents="Give me a recipe for chocolate cake.",
            config={
                "response_mime_type": "application/json",
                "response_schema": Recipe,
            }
        )
        
        # Parse result directly into Pydantic model
        recipe = response.parsed
        print(f"Title: {recipe.title}")
        print(f"Ingredients: {recipe.ingredients}")
        
    except Exception as e:
        print(f"Error: {e}")
```

#### 4. LangSmith Tracing Integration (Optional)
If you have `langsmith` installed, `gemini-rotate` automatically wraps the internal Google GenAI clients using LangSmith's standard Gemini wrapper. This enables automatic tracing of your generated content requests.

To activate tracing, configure your environment with the standard LangSmith variables:
```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-api-key"
# Optional:
export LANGCHAIN_PROJECT="my-gemini-project"
```

By default, the library traces requests with the tag `"gemini-rotate"` and integration metadata. You can customize tags, metadata, or extra parameters by passing `tracing_extra` to the `GeminiRotationClient` constructor:
```python
import asyncio
from gemini_rotate import GeminiRotationClient

async def main():
    # Initialize client with custom tracing tags and metadata
    client = GeminiRotationClient(
        tracing_extra={
            "tags": ["production", "chatbot"],
            "metadata": {
                "user_id": "user_123"
            }
        }
    )
    
    response = await client.generate_content("Describe neural networks in one sentence.")
    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `tracing_extra` | `dict` | (Optional) Extra tracing configuration (tags, metadata) passed to LangSmith's wrapping utility. |
| `contents` | `str` \| `list` | The prompt or content to send. |
| `config` | `dict` | (Optional) Generation config (temperature, tools, schema) passed to `google.genai`. |

## ⚙️ Configuration

### 1. The `.env` Format (Expected)
To configure the library, copy the provided [.env.example](file:///Users/jayed/gits/gemini-rotate/.env.example) file to `.env` in the root of your project.

The environment configuration supports the following parameters:

```env
# Required: Define your Gemini API keys using the pattern GEMINI_API_KEY_<number>
GEMINI_API_KEY_1="AIzaSy..."
GEMINI_API_KEY_2="AI3yhj..."
GEMINI_API_KEY_3="AIdf56..."

# Optional: Define your models in a valid JSON array format.
# The models will be processed in pairs (Primary -> Secondary fallback).
GEMINI_MODELS='["gemini-3.5-flash", "gemini-3.1-pro-preview"]'

# Optional: LangSmith Tracing integration config.
# Note: Leaving these variables empty (e.g. LANGCHAIN_TRACING_V2=) or unconfigured
# will automatically disable tracing wrapper logic without throwing missing key errors.
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="gemini-rotate"
```

*(Note: A single `GEMINI_API_KEY` environment variable is also supported as a fallback. If numbered keys like `GEMINI_API_KEY_1` are present, they are guaranteed to rotate in sequential order.)*

### 2. Model Priority Breakdown
You can customize the order in which models are attempted by setting `GEMINI_MODELS` in `.env` as shown above. The string MUST be a valid JSON array. The library processes models in **Primary -> Secondary** pairs.

**Default Behavior (if GEMINI_MODELS is not set):**
1.  `gemini-3.5-flash` -> `gemini-3.1-pro-preview`
2.  `gemini-3.1-flash-lite` -> `gemini-3.1-flash-lite-preview`
3.  `gemini-3-flash-preview` -> `gemini-2.5-flash`
4.  `gemini-2.5-flash-lite` -> `gemini-2.5-pro`
5.  `gemini-flash-latest` -> `gemini-flash-lite-latest`
6.  `gemma-4-26b-a4b-it` -> `gemma-4-31b-it`

**Custom Configuration:**
```env
GEMINI_MODELS='["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]'
```

## 🔍 How it Works

### 1. Execution Flow & Retries
The library attempts model pairs sequentially using rotated API clients:

```mermaid
graph TD
    Start[Start Request] --> IsTraced{LangSmith Enabled?}
    IsTraced -->|Yes| StartTrace[Start Trace Parent Run]
    IsTraced -->|No| LoopPairs
    StartTrace --> LoopPairs{Loop Model Pairs}
    
    LoopPairs -->|Primary, Secondary| LoopClients{Loop API Clients}
    
    LoopClients -->|Next Client| AttemptPrimary[Attempt Primary Model]
    
    AttemptPrimary -->|Success| LogTraceSuccess[Log Client/Model Success & Cost to Trace]
    AttemptPrimary -->|Failure| CheckSecondary{Has Secondary Model?}
    
    CheckSecondary -->|Yes| AttemptSecondary[Attempt Secondary Model]
    CheckSecondary -->|No| NextClient[Next Client]
    
    AttemptSecondary -->|Success| LogTraceSuccess
    AttemptSecondary -->|Failure| NextClient
    
    LogTraceSuccess --> ReturnResponse[Return Response]
    
    NextClient -->|Clients Exhausted| NextPair[Next Pair]
    NextPair -->|Pairs Exhausted| LogTraceFailure[Log Failure to Trace]
    LogTraceFailure --> RaiseError[Raise AllClientsFailed]
    
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style ReturnResponse fill:#9f9,stroke:#333,stroke-width:2px
    style RaiseError fill:#f99,stroke:#333,stroke-width:2px
```

### 2. Tracing Pipeline Integration
If LangSmith tracing is enabled:
* **Parent Trace Context**: The entire rotation execution is wrapped in a high-level parent run, allowing you to see the aggregate latency and outcome.
* **Rotated Client Wrapping**: Each raw client inside the pool gets its own metadata identifiers (`Client-1`, `Client-2`, etc.) matching their corresponding API keys.
* **Outcome Attribution**: The successfully resolved key and model are dynamically logged under the trace's metadata attributes (`succeeded_client`, `succeeded_model`, `ls_model_name`).
* **Precise Cost & Token Tracking**: The library dynamically calculates the precise model usage pricing based on the official guidelines (including dynamic token tiers) and posts token counts and the calculated total USD cost directly to the trace outputs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
