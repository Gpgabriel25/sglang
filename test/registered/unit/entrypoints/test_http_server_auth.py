"""Exercise auth on the imported HTTP app without model or scheduler startup."""

import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@pytest.fixture
def worker_app(monkeypatch):
    app = http_server.app
    serving = None

    def get_serving():
        assert serving is not None, "Worker configuration has not been published"
        return serving

    @asynccontextmanager
    async def lifespan(app):
        nonlocal serving
        assert app.middleware_stack is not None
        serving = SimpleNamespace(api_key="user", admin_api_key="admin")
        yield

    manager = SimpleNamespace(
        served_model_name="auth-fixture",
        model_config=SimpleNamespace(context_len=4096),
        flush_cache=AsyncMock(return_value=SimpleNamespace(success=True)),
    )
    monkeypatch.setattr(http_server, "get_serving", get_serving)
    monkeypatch.setattr(
        http_server, "get_lora", lambda: SimpleNamespace(enable_lora=False)
    )
    monkeypatch.setattr(
        http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
    )
    monkeypatch.setattr(app.router, "lifespan_context", lifespan)
    monkeypatch.setattr(app, "middleware_stack", None)
    return app, manager


def test_reimported_app_authenticates_models_and_admin_routes(worker_app):
    app, manager = worker_app
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        result = client.get("/v1/models", headers={"Authorization": "Bearer user"})
        assert result.status_code == 200
        assert result.json()["data"][0]["id"] == "auth-fixture"
        assert (
            client.post(
                "/flush_cache", headers={"Authorization": "Bearer user"}
            ).status_code
            == 401
        )
        manager.flush_cache.assert_not_awaited()
        assert (
            client.post(
                "/flush_cache", headers={"Authorization": "Bearer admin"}
            ).status_code
            == 200
        )


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_reimported_app_authenticates_streaming_requests(worker_app, monkeypatch, path):
    app, _ = worker_app

    async def events():
        yield "data: first\n\n"
        yield "data: [DONE]\n\n"

    async def chat(request, raw_request):
        return StreamingResponse(events(), media_type="text/event-stream")

    async def responses(request, raw_request):
        return events()

    handler = AsyncMock(side_effect=chat if path.endswith("completions") else responses)
    state_name = (
        "openai_serving_chat"
        if path.endswith("completions")
        else "openai_serving_responses"
    )
    serving = SimpleNamespace(handle_request=handler, create_responses=handler)
    monkeypatch.setattr(app.state, state_name, serving, raising=False)
    payload = {"model": "auth-fixture", "stream": True}
    if path.endswith("completions"):
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["input"] = "hello"

    with TestClient(app) as client:
        for token in (None, "wrong"):
            headers = {} if token is None else {"Authorization": f"Bearer {token}"}
            assert client.post(path, json=payload, headers=headers).status_code == 401
        handler.assert_not_awaited()
        response = client.post(
            path, json=payload, headers={"Authorization": "Bearer user"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text == "data: first\n\ndata: [DONE]\n\n"


def test_multi_tokenizer_initialization_accepts_configured_keys(monkeypatch, tmp_path):
    args = SimpleNamespace(
        api_key="user", admin_api_key="admin", tokenizer_worker_num=4
    )
    port_args = SimpleNamespace()
    info = {"max_req_input_len": 4096, "startup_time": {}}
    manager = Mock()
    monkeypatch.setattr(
        http_server, "read_from_shared_memory", lambda name: (port_args, args, info)
    )
    monkeypatch.setattr(http_server, "publish", lambda *a, **kw: None)
    monkeypatch.setattr(
        http_server,
        "get_serving",
        lambda: SimpleNamespace(
            **vars(args), chat_template=None, completion_template=None
        ),
    )
    monkeypatch.setattr(
        http_server, "get_model", lambda: SimpleNamespace(model_path="fixture")
    )
    monkeypatch.setattr(
        http_server, "get_tokenizer_worker_class", lambda args: lambda *a: manager
    )
    monkeypatch.setattr(http_server, "TemplateManager", Mock)
    monkeypatch.setattr(http_server.setproctitle, "setproctitle", lambda title: None)
    monkeypatch.setattr(http_server.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(
        http_server.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(name=str(tmp_path / "worker.ipc")),
    )
    # Restore the process-global manager after initialization.
    monkeypatch.setattr(http_server, "_global_state", None)
    result = asyncio.run(http_server.init_multi_tokenizer())
    assert result is args
    assert http_server.get_global_state().tokenizer_manager is manager
    assert manager.max_req_input_len == 4096


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
