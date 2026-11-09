"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import json

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_make_quiz_is_registered_and_called():
    model = TestModel(call_tools=["make_quiz"])
    with agent.override(model=model):
        result = agent.run_sync("第2回の確認問題を出して")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "make_quiz" in offered
    calls = [p for p in _parts(result, ToolCallPart) if p.tool_name == "make_quiz"]
    returns = [p for p in _parts(result, ToolReturnPart) if p.tool_name == "make_quiz"]
    assert calls and returns


def test_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch):
    from app import agent as agent_module, tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"], custom_output_text="完了")
    monkeypatch.setattr(agent_module, "build_model", lambda: model)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr("sys.argv", ["app.agent", "このやりとりを保存して"])

    agent_module.main()

    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    saved_messages = json.loads(logs[0].read_text(encoding="utf-8"))
    assert "このやりとりを保存して" in logs[0].read_text(encoding="utf-8")
    assert "save_learning_log" in logs[0].read_text(encoding="utf-8")
    assert saved_messages


def test_learning_log_is_not_saved_when_approval_is_denied(tmp_path, monkeypatch):
    from app import agent as agent_module, tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"], custom_output_text="中止しました")
    monkeypatch.setattr(agent_module, "build_model", lambda: model)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr("sys.argv", ["app.agent", "このやりとりを保存して"])

    agent_module.main()

    assert not list(tmp_path.iterdir())
