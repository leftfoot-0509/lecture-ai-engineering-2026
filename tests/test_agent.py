"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import json

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import INSTRUCTIONS, agent

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
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
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


def test_learning_log_is_saved_only_after_post_response_approval(
    tmp_path, monkeypatch, capsys
):
    from app import agent as agent_module, tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(
        call_tools=["search_glossary", "save_learning_log"],
        custom_output_text="完了",
    )
    monkeypatch.setattr(agent_module, "build_model", lambda: model)

    def approve_after_response(prompt):
        assert "完了" in capsys.readouterr().out
        assert "保存しますか" in prompt
        return "y"

    monkeypatch.setattr("builtins.input", approve_after_response)
    monkeypatch.setattr("sys.argv", ["app.agent", "RAGとは？"])

    agent_module.main()

    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    saved_log = json.loads(logs[0].read_text(encoding="utf-8"))
    assert set(saved_log) == {"timestamp", "user_input", "llm_answer", "tool return"}
    assert saved_log["user_input"] == "RAGとは？"
    assert saved_log["llm_answer"] == "完了"
    assert saved_log["timestamp"] == logs[0].stem
    assert len(saved_log["timestamp"]) == 15
    assert saved_log["timestamp"][8] == "T"
    assert saved_log["timestamp"].replace("T", "").isdigit()
    assert [item["tool_name"] for item in saved_log["tool return"]] == [
        "search_glossary"
    ]


def test_instructions_call_save_tool_after_other_tools_every_time():
    assert (
        "利用者が保存を頼んだかどうかにかかわらず、毎回 save_learning_log を呼ぶ。"
        in INSTRUCTIONS
    )
    assert (
        "save_learning_log は、必要な他のツールをすべて使い終えた後、"
        "回答前の最後のツール呼び出しとして使う。"
        in INSTRUCTIONS
    )


def test_learning_log_is_not_saved_when_post_response_approval_is_denied(
    tmp_path, monkeypatch, capsys
):
    from app import agent as agent_module, tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    model = TestModel(call_tools=[], custom_output_text="中止しました")
    monkeypatch.setattr(agent_module, "build_model", lambda: model)

    def deny_after_response(prompt):
        assert "中止しました" in capsys.readouterr().out
        assert "保存しますか" in prompt
        return "n"

    monkeypatch.setattr("builtins.input", deny_after_response)
    monkeypatch.setattr("sys.argv", ["app.agent", "このやりとりを保存して"])

    agent_module.main()

    assert not list(tmp_path.iterdir())
