"""
OpenRouter model configuration for LLM-as-judge scripts.

Models organized by use case and cost tier.
Updated: January 2026
"""

# OpenRouter API settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/relationship-advice-analysis",
    "X-Title": "Relationship Advice Analysis"
}

# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODELS = {
    # -------------------------------------------------------------------------
    # FREE TIER (rate limited, great for testing)
    # -------------------------------------------------------------------------
    "deepseek-r1-free": {
        "id": "deepseek/deepseek-r1-0528:free",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "context": 163000,
        "reasoning": True,
        "description": "DeepSeek R1 FREE - reasoning model, rate limited"
    },
    "gemini-flash-free": {
        "id": "google/gemini-2.0-flash-exp:free",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 2.0 Flash FREE - 1M context, rate limited"
    },
    "llama-405b-free": {
        "id": "meta-llama/llama-3.1-405b-instruct:free",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "context": 131000,
        "reasoning": False,
        "description": "Llama 3.1 405B FREE - largest open model, rate limited"
    },

    # -------------------------------------------------------------------------
    # BUDGET REASONING (with thinking traces)
    # -------------------------------------------------------------------------
    "deepseek-r1-distill": {
        "id": "deepseek/deepseek-r1-distill-llama-70b",
        "input_cost": 0.03,
        "output_cost": 0.11,
        "context": 131000,
        "reasoning": True,
        "description": "DeepSeek R1 distilled to Llama 70B - cheapest reasoning"
    },
    "qwen3-thinking": {
        "id": "qwen/qwen3-30b-a3b-thinking-2507",
        "input_cost": 0.051,
        "output_cost": 0.34,
        "context": 32000,
        "reasoning": True,
        "description": "Qwen3 30B Thinking - July 2025, affordable reasoning"
    },
    "phi-4-reasoning": {
        "id": "microsoft/phi-4-reasoning-plus",
        "input_cost": 0.07,
        "output_cost": 0.35,
        "context": 32000,
        "reasoning": True,
        "description": "Microsoft Phi-4 Reasoning Plus - compact reasoning"
    },
    "qwen3-235b": {
        "id": "qwen/qwen3-235b-a22b-2507",
        "input_cost": 0.071,
        "output_cost": 0.463,
        "context": 262000,
        "reasoning": False,
        "description": "Qwen3 235B Instruct - large MoE model, no thinking"
    },
    "qwen3-235b-thinking": {
        "id": "qwen/qwen3-235b-a22b-thinking-2507",
        "input_cost": 0.11,
        "output_cost": 0.60,
        "context": 262000,
        "reasoning": True,
        "description": "Qwen3 235B Thinking - large reasoning model"
    },
    "qwq-32b": {
        "id": "qwen/qwq-32b",
        "input_cost": 0.15,
        "output_cost": 0.40,
        "context": 32000,
        "reasoning": True,
        "description": "Qwen QwQ 32B - established reasoning model"
    },

    # -------------------------------------------------------------------------
    # BUDGET GENERATION (under $0.30/M input)
    # -------------------------------------------------------------------------
    "gpt-5-nano": {
        "id": "openai/gpt-5-nano",
        "input_cost": 0.05,
        "output_cost": 0.40,
        "context": 400000,
        "reasoning": False,
        "description": "GPT-5 Nano - cheapest GPT-5, 400k context"
    },
    "gemini-flash-lite": {
        "id": "google/gemini-2.0-flash-lite-001",
        "input_cost": 0.075,
        "output_cost": 0.30,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 2.0 Flash Lite - cheapest Gemini, 1M context"
    },
    "gpt-4.1-nano": {
        "id": "openai/gpt-4.1-nano",
        "input_cost": 0.10,
        "output_cost": 0.40,
        "context": 1047000,
        "reasoning": False,
        "description": "GPT-4.1 Nano - 1M context, very cheap"
    },
    "gemini-flash": {
        "id": "google/gemini-2.0-flash-001",
        "input_cost": 0.10,
        "output_cost": 0.40,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 2.0 Flash - fast, 1M context"
    },
    "gemini-2.5-flash-lite": {
        "id": "google/gemini-2.5-flash-lite",
        "input_cost": 0.10,
        "output_cost": 0.40,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 2.5 Flash Lite - latest lite, 1M context"
    },
    "gpt-4o-mini": {
        "id": "openai/gpt-4o-mini",
        "input_cost": 0.15,
        "output_cost": 0.60,
        "context": 128000,
        "reasoning": False,
        "description": "GPT-4o Mini - reliable OpenAI option"
    },
    "deepseek-v3.1": {
        "id": "deepseek/deepseek-chat-v3.1",
        "input_cost": 0.15,
        "output_cost": 0.75,
        "context": 32000,
        "reasoning": False,
        "description": "DeepSeek Chat V3.1 - latest chat model"
    },
    "claude-3-haiku": {
        "id": "anthropic/claude-3-haiku",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "context": 200000,
        "reasoning": False,
        "description": "Claude 3 Haiku - fast, affordable Claude"
    },
    "deepseek-v3.2": {
        "id": "deepseek/deepseek-v3.2",
        "input_cost": 0.25,
        "output_cost": 0.38,
        "context": 163000,
        "reasoning": False,
        "description": "DeepSeek V3.2 - latest DeepSeek, 163k context"
    },
    "gpt-5-mini": {
        "id": "openai/gpt-5-mini",
        "input_cost": 0.25,
        "output_cost": 2.00,
        "context": 400000,
        "reasoning": False,
        "description": "GPT-5 Mini - smaller GPT-5, 400k context"
    },

    # -------------------------------------------------------------------------
    # MID-TIER ($0.30 - $1.00/M input)
    # -------------------------------------------------------------------------
    "gemini-2.5-flash": {
        "id": "google/gemini-2.5-flash",
        "input_cost": 0.30,
        "output_cost": 2.50,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 2.5 Flash - latest Gemini Flash, 1M context"
    },
    "kimi-thinking": {
        "id": "moonshotai/kimi-k2-thinking",
        "input_cost": 0.40,
        "output_cost": 1.75,
        "context": 262000,
        "reasoning": True,
        "description": "Kimi K2 Thinking - Moonshot reasoning, 262k context"
    },
    "gpt-4.1-mini": {
        "id": "openai/gpt-4.1-mini",
        "input_cost": 0.40,
        "output_cost": 1.60,
        "context": 1047000,
        "reasoning": False,
        "description": "GPT-4.1 Mini - 1M context mid-tier"
    },
    "deepseek-r1": {
        "id": "deepseek/deepseek-r1-0528",
        "input_cost": 0.45,
        "output_cost": 2.15,
        "context": 131000,
        "reasoning": True,
        "description": "DeepSeek R1 (May 2028) - full reasoning model"
    },
    "gemini-3-flash": {
        "id": "google/gemini-3-flash-preview",
        "input_cost": 0.50,
        "output_cost": 3.00,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 3 Flash Preview - latest Gemini generation"
    },
    "mistral-large": {
        "id": "mistralai/mistral-large-2512",
        "input_cost": 0.50,
        "output_cost": 1.50,
        "context": 128000,
        "reasoning": False,
        "description": "Mistral Large 2512 - European flagship model"
    },
    "gemini-3-pro": {
        "id": "google/gemini-3-pro-preview",
        "input_cost": 2.00,
        "output_cost": 12.00,
        "context": 1048000,
        "reasoning": False,
        "description": "Gemini 3 Pro Preview - highest quality Gemini"
    },
    "deepseek-r1-latest": {
        "id": "deepseek/deepseek-r1",
        "input_cost": 0.70,
        "output_cost": 2.40,
        "context": 163000,
        "reasoning": True,
        "description": "DeepSeek R1 Latest - best DeepSeek reasoning"
    },
    "claude-3.5-haiku": {
        "id": "anthropic/claude-3.5-haiku",
        "input_cost": 0.80,
        "output_cost": 4.00,
        "context": 200000,
        "reasoning": False,
        "description": "Claude 3.5 Haiku - fast, high quality"
    },
    "claude-haiku-4.5": {
        "id": "anthropic/claude-haiku-4.5",
        "input_cost": 1.00,
        "output_cost": 5.00,
        "context": 200000,
        "reasoning": False,
        "description": "Claude Haiku 4.5 - latest Haiku"
    },

    # -------------------------------------------------------------------------
    # PREMIUM REASONING ($1.00+/M input)
    # -------------------------------------------------------------------------
    "o3-mini": {
        "id": "openai/o3-mini",
        "input_cost": 1.10,
        "output_cost": 4.40,
        "context": 200000,
        "reasoning": True,
        "description": "OpenAI O3 Mini - OpenAI reasoning"
    },
    "o4-mini": {
        "id": "openai/o4-mini",
        "input_cost": 1.10,
        "output_cost": 4.40,
        "context": 200000,
        "reasoning": True,
        "description": "OpenAI O4 Mini - latest OpenAI reasoning"
    },
    "o3": {
        "id": "openai/o3",
        "input_cost": 2.00,
        "output_cost": 8.00,
        "context": 200000,
        "reasoning": True,
        "description": "OpenAI O3 - full reasoning model"
    },

    # -------------------------------------------------------------------------
    # FRONTIER MODELS (highest quality)
    # -------------------------------------------------------------------------
    "gpt-5": {
        "id": "openai/gpt-5",
        "input_cost": 1.25,
        "output_cost": 10.00,
        "context": 400000,
        "reasoning": False,
        "description": "GPT-5 - OpenAI flagship, 400k context"
    },
    "gpt-5.1": {
        "id": "openai/gpt-5.1",
        "input_cost": 1.25,
        "output_cost": 10.00,
        "context": 400000,
        "reasoning": False,
        "description": "GPT-5.1 - latest GPT, 400k context"
    },
    "gpt-5.2-codex": {
        "id": "openai/gpt-5.2-codex",
        "input_cost": 1.75,
        "output_cost": 14.00,
        "context": 400000,
        "reasoning": False,
        "description": "GPT-5.2 Codex - newest GPT, code-optimized"
    },
    "gpt-5.2-chat": {
        "id": "openai/gpt-5.2-chat",
        "input_cost": 1.75,
        "output_cost": 14.00,
        "context": 128000,
        "reasoning": False,
        "description": "GPT-5.2 Chat - newest GPT chat model"
    },
    "grok-4.1-fast": {
        "id": "x-ai/grok-4.1-fast",
        "input_cost": 0.20,
        "output_cost": 0.50,
        "context": 2000000,
        "reasoning": False,
        "description": "Grok 4.1 Fast - xAI, 2M context, very cheap"
    },
    "claude-sonnet-4": {
        "id": "anthropic/claude-sonnet-4",
        "input_cost": 3.00,
        "output_cost": 15.00,
        "context": 1000000,
        "reasoning": False,
        "description": "Claude Sonnet 4 - 1M context flagship"
    },
    "claude-sonnet-4.5": {
        "id": "anthropic/claude-sonnet-4.5",
        "input_cost": 3.00,
        "output_cost": 15.00,
        "context": 1000000,
        "reasoning": False,
        "description": "Claude Sonnet 4.5 - latest Sonnet, 1M context"
    },
    "claude-sonnet-4.5-thinking": {
        "id": "anthropic/claude-sonnet-4.5",
        "input_cost": 3.00,
        "output_cost": 15.00,
        "context": 1000000,
        "reasoning": True,
        "reasoning_param": True,
        "description": "Claude Sonnet 4.5 with extended thinking"
    },
    "gemini-3-flash-thinking": {
        "id": "google/gemini-3-flash-preview",
        "input_cost": 0.50,
        "output_cost": 3.00,
        "context": 1048000,
        "reasoning": True,
        "reasoning_param": True,
        "description": "Gemini 3 Flash with reasoning mode"
    },
    "claude-opus-4.5": {
        "id": "anthropic/claude-opus-4.5",
        "input_cost": 5.00,
        "output_cost": 25.00,
        "context": 200000,
        "reasoning": False,
        "description": "Claude Opus 4.5 - high quality Opus"
    },
    "claude-opus-4": {
        "id": "anthropic/claude-opus-4",
        "input_cost": 15.00,
        "output_cost": 75.00,
        "context": 200000,
        "reasoning": False,
        "description": "Claude Opus 4 - top-tier Claude"
    },
}

# Default models for each task
DEFAULTS = {
    "generate": "deepseek-v3.2",        # Best value for generation
    "analyze": "qwen3-235b-thinking",   # Affordable reasoning with traces
    "analyze_premium": "deepseek-r1",   # Better reasoning
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model(name: str) -> dict:
    """Get model config by short name."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]


def list_models(reasoning_only: bool = False, free_only: bool = False) -> None:
    """Print available models with pricing."""
    print("\nAvailable Models:")
    print("-" * 100)
    print(f"{'Name':<28} {'Model ID':<45} {'$/M In':<10} {'Reason':<8}")
    print("-" * 100)

    for name, cfg in sorted(MODELS.items(), key=lambda x: x[1]["input_cost"]):
        if reasoning_only and not cfg["reasoning"]:
            continue
        if free_only and cfg["input_cost"] > 0:
            continue
        reasoning = "✓" if cfg["reasoning"] else ""
        cost = "FREE" if cfg["input_cost"] == 0 else f"${cfg['input_cost']:.2f}"
        print(f"{name:<28} {cfg['id']:<45} {cost:<10} {reasoning:<8}")
    print()


def estimate_cost(
    n_posts: int,
    input_tokens_per_post: int,
    output_tokens_per_post: int,
    model_name: str
) -> float:
    """Estimate cost for a run."""
    model = get_model(model_name)
    input_cost = (n_posts * input_tokens_per_post / 1_000_000) * model["input_cost"]
    output_cost = (n_posts * output_tokens_per_post / 1_000_000) * model["output_cost"]
    return input_cost + output_cost


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("FREE MODELS (rate limited, great for testing)")
    print("=" * 100)
    list_models(free_only=True)

    print("\n" + "=" * 100)
    print("REASONING MODELS (with thinking traces)")
    print("=" * 100)
    list_models(reasoning_only=True)

    print("\n" + "=" * 100)
    print("ALL MODELS")
    print("=" * 100)
    list_models()

    print("\n" + "=" * 100)
    print("COST ESTIMATES (32,600 posts)")
    print("=" * 100)
    estimates = [
        ("generate", 2000, 400, "gpt-5-nano"),
        ("generate", 2000, 400, "deepseek-v3.2"),
        ("generate", 2000, 400, "gemini-2.5-flash"),
        ("generate", 2000, 400, "claude-3-haiku"),
        ("analyze", 3000, 800, "deepseek-r1-distill"),
        ("analyze", 3000, 800, "qwen3-thinking"),
        ("analyze", 3000, 800, "deepseek-r1"),
        ("analyze", 3000, 800, "claude-sonnet-4.5-thinking"),
    ]
    for task, inp, out, model in estimates:
        cost = estimate_cost(32600, inp, out, model)
        print(f"  {task:12} with {model:<30} ${cost:>8.2f}")
