from types import SimpleNamespace

from app import ask as ask_module


class FakeChat:
    def __init__(self, responses, before_send=None):
        self.responses = iter(responses)
        self.messages = []
        self.before_send = before_send

    def send_message(self, message):
        if self.before_send:
            self.before_send()
        self.messages.append(message)
        return next(self.responses)


class FakeClient:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.closed = True


def test_ask_sends_question_and_prints_token_usage(capsys):
    chat = FakeChat(
        [
            SimpleNamespace(
                text="回答です",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=8,
                ),
            )
        ]
    )

    assert ask_module.ask("最初の質問", chat) == "回答です"

    assert chat.messages == ["最初の質問"]
    assert "[tokens] 入力=12, 出力=8" in capsys.readouterr().out


def test_main_keeps_chat_history_until_finish(monkeypatch, capsys):
    client = FakeClient()

    def assert_client_open():
        assert not client.closed

    chat = FakeChat(
        [
            SimpleNamespace(text="最初の回答", usage_metadata=None),
            SimpleNamespace(text="続きの回答", usage_metadata=None),
        ],
        before_send=assert_client_open,
    )
    questions = iter(["続けて質問", " finish "])

    def make_chat(active_client):
        assert active_client is client
        return chat

    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(ask_module, "create_chat", make_chat)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])

    ask_module.main()

    output = capsys.readouterr().out
    assert client.closed
    assert chat.messages == ["最初の質問", "続けて質問"]
    assert "最初の回答" in output
    assert "続きの回答" in output