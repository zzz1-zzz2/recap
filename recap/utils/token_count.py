"""Token counting utilities."""
import re

# GPT-4 / DeepSeek approximate tokenizer (chars per token)
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate token count (chars / 4)."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def count_message_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for m in messages:
        content = str(m.get("content", "") or "")
        total += estimate_tokens(content)
    return total


def truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    budget_chars = int(budget_tokens * _CHARS_PER_TOKEN)
    if len(text) <= budget_chars:
        return text
    return text[:budget_chars] + "\n... (truncated)"
