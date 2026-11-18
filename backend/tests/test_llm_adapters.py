from __future__ import annotations

from stats_analyst.config import Settings
from stats_analyst.llm import ChatMessage, ToolCall, create_llm_client
from stats_analyst.llm.anthropic_client import _to_anthropic
from stats_analyst.llm.ollama_client import _ensure_dict, _to_ollama


def test_factory_switches_provider() -> None:
    assert create_llm_client(Settings(llm_provider="ollama")).provider == "ollama"
    client = create_llm_client(Settings(llm_provider="anthropic", anthropic_api_key="test"))
    assert client.provider == "anthropic"
    assert client.model == "claude-haiku-5-5"


def test_anthropic_merges_parallel_tool_results_into_one_user_turn() -> None:
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall("1", "a", {}), ToolCall("2", "b", {})],
        ),
        ChatMessage(role="tool", content="{}", tool_call_id="1", tool_name="a"),
        ChatMessage(role="tool", content="{}", tool_call_id="2", tool_name="b", is_error=True),
    ]
    converted = _to_anthropic(history)
    assert [turn["role"] for turn in converted] == ["user", "assistant", "user"]
    assert [block["tool_use_id"] for block in converted[2]["content"]] == ["1", "2"]
    assert converted[2]["content"][1]["is_error"] is True


def test_anthropic_replays_raw_assistant_content() -> None:
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "x"}]
    converted = _to_anthropic([ChatMessage(role="assistant", content="x", raw=raw)])
    assert converted[0]["content"] is raw


def test_ollama_message_conversion() -> None:
    tool_msg = ChatMessage(role="tool", content="{}", tool_call_id="1", tool_name="describe")
    assert _to_ollama(tool_msg) == {"role": "tool", "content": "{}", "tool_name": "describe"}
    assistant = ChatMessage(role="assistant", tool_calls=[ToolCall("1", "f", {"x": 1})])
    assert _to_ollama(assistant)["tool_calls"] == [
        {"function": {"name": "f", "arguments": {"x": 1}}}
    ]
    assert _ensure_dict('{"a": 1}') == {"a": 1}
    assert _ensure_dict("not json") == {}
