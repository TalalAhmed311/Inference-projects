"""OpenAI-compatible API tests (FastAPI TestClient, real small model)."""

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def client(engine):
    from tiny_engine.serving import build_app

    with TestClient(build_app(engine)) as c:  # runs the lifespan: starts/stops the engine thread
        yield c


def sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_health_version_models(client, engine):
    assert client.get("/health").status_code == 200
    assert client.get("/version").json()["version"].startswith("tiny_engine")
    models = client.get("/v1/models").json()["data"]
    assert models[0]["id"] == engine.model_name
    assert models[0]["max_model_len"] == engine.max_model_len


def test_chat_completion(client, engine):
    r = client.post("/v1/chat/completions", json={
        "model": engine.model_name, "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16, "temperature": 0})
    body = r.json()
    assert r.status_code == 200, body
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] <= 16


def test_chat_stream_with_usage(client):
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 24,
            "stream": True, "stream_options": {"include_usage": True}}) as r:
        assert r.status_code == 200
        events = sse_events(r)
    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e.get("choices"))
    assert content
    usage = events[-2]
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] > 0


def test_completion_ignore_eos_exact_length(client):
    r = client.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 20, "ignore_eos": True})
    assert r.json()["usage"]["completion_tokens"] == 20


def test_completion_stream(client):
    with client.stream("POST", "/v1/completions", json={"prompt": "The sky is", "max_tokens": 8, "stream": True}) as r:
        events = sse_events(r)
    assert events[-1] == "[DONE]"
    assert events[-2]["choices"][0]["finish_reason"] in ("stop", "length")


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "tiny:num_requests_running" in text and "tiny:generation_tokens_total" in text


def test_errors(client, engine):
    assert client.post("/v1/completions", json={"model": "nope", "prompt": "x"}).status_code == 404
    too_long = client.post("/v1/completions", json={"prompt": [1] * engine.max_model_len})
    assert too_long.status_code == 400 and "maximum context length" in too_long.json()["error"]["message"]
    assert client.post("/v1/completions", json={"prompt": "x", "max_tokens": 4, "n": 2}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "x", "top_p": 0}).status_code == 400
