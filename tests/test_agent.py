"""Ferramenta SQL auditada, modo offline e protocolo de tool calling."""
from types import SimpleNamespace

import duckdb
import pytest

from src import agent


def test_sql_rejects_arbitrary_input(tmp_path) -> None:
    """SQL arbitrário não alcança o banco."""
    with pytest.raises(ValueError):
        agent.query_database("DROP TABLE accounts", tmp_path / "unused.duckdb")


def test_sql_reads_evidence(tmp_path) -> None:
    """Consulta auditada retorna números reais do banco de teste."""
    database = tmp_path / "test.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE transactions(amount DOUBLE)")
        connection.execute("INSERT INTO transactions VALUES (10), (30)")
    result = agent.query_database("executivo", database)
    assert result["rows"][0]["average_ticket"] == 20
    assert "SELECT" in result["sql"]


def test_offline_without_key(monkeypatch) -> None:
    """Ausência de chave mantém a funcionalidade sem acesso à rede."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "query_database", lambda report: {"report": report, "rows": []})
    result = agent.explain("churn", use_api=True)
    assert result["mode"] == "offline"
    assert "registros" in result["answer"]


def test_api_tool_roundtrip(monkeypatch) -> None:
    """Simula a chamada de ferramenta e a resposta final sem consumir API."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    evidence = {"report": "executivo", "rows": [], "sql": "SELECT 1"}
    monkeypatch.setattr(agent, "query_database", lambda report: evidence)
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="query_database", arguments='{"report":"executivo"}'))
    first = SimpleNamespace(tool_calls=[call], model_dump=lambda **kwargs: {"role": "assistant", "content": None})
    last = SimpleNamespace(tool_calls=None, content="Explicação baseada no SQL.", model_dump=lambda **kwargs: {"role": "assistant", "content": "Explicação"})
    messages = iter([first, last])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=next(messages))]))))
    monkeypatch.setattr(agent, "OpenAI", lambda **kwargs: client)
    result = agent.explain("resumo", use_api=True)
    assert result["mode"] == "openai"
    assert result["evidence"] == [evidence]
