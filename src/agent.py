"""Agente com ferramenta SQL restrita, evidências auditáveis e fallback offline."""
import json
import os
from pathlib import Path

import duckdb
from openai import OpenAI

from src.config import DATA

SQL = {
    "executivo": "SELECT count(*) AS transactions, round(sum(amount), 2) AS total_amount, round(avg(amount), 2) AS average_ticket FROM transactions",
    "anomalias": "SELECT transaction_id, account_id, date, amount, hour, anomaly_score FROM anomaly_cases ORDER BY anomaly_score DESC LIMIT 5",
    "churn": "SELECT account_id, churn_probability, recency, count_recent, count_previous, activity_ratio FROM churn_predictions ORDER BY churn_probability DESC LIMIT 5",
}


def query_database(report: str, database: Path = DATA / "analytics.duckdb") -> dict:
    """Executa somente uma das consultas SELECT auditadas; não aceita SQL arbitrário."""
    if report not in SQL:
        raise ValueError("Relatório não permitido.")
    with duckdb.connect(str(database), read_only=True, config={"enable_external_access": False}) as connection:
        rows = connection.execute(SQL[report]).df().to_dict("records")
    return {"report": report, "sql": SQL[report], "rows": rows}


def offline_explanation(evidence: dict) -> str:
    """Converte evidências em narrativa determinística sem alegar causalidade."""
    rows = evidence["rows"]
    if not rows:
        return "Não há registros disponíveis para este relatório."
    if evidence["report"] == "churn":
        row = rows[0]
        return (f"A conta {row['account_id']} tem probabilidade prevista de inatividade "
                f"em 30 dias de {row['churn_probability']:.1%}. "
                f"Realizou {row['count_recent']:.0f} transações nos últimos 30 dias, "
                f"ante {row['count_previous']:.0f} na janela anterior, e está há "
                f"{row['recency']:.0f} dias sem transacionar. São sinais associados ao risco; "
                "não provam a causa do abandono. O ranking cobre somente contas do teste reservado.")
    if evidence["report"] == "anomalias":
        row = rows[0]
        return (f"A transação {row['transaction_id']} da conta {row['account_id']} lidera "
                f"o ranking de anomalias: R$ {row['amount']:.2f}, às {row['hour']}h, "
                f"score {row['anomaly_score']:.3f}. O detector considera valor, horário "
                "e dia da semana. Um score elevado exige revisão; não comprova fraude.")
    row = rows[0]
    return (f"A base contém {row['transactions']:,} transações e R$ {row['total_amount']:,.2f} "
            f"movimentados, com ticket médio de R$ {row['average_ticket']:.2f}.")


def explain(question: str, use_api: bool = False) -> dict:
    """Responde offline por padrão ou usa até três rodadas de ferramentas OpenAI."""
    lowered = question.lower()
    report = "churn" if any(word in lowered for word in ("churn", "inativ", "abandono")) else (
        "anomalias" if any(word in lowered for word in ("anom", "fraud")) else "executivo")
    evidence = query_database(report)
    fallback = {"mode": "offline", "answer": offline_explanation(evidence), "evidence": [evidence]}
    if not use_api or not os.getenv("OPENAI_API_KEY"):
        return fallback
    tool = {"type": "function", "function": {
        "name": "query_database", "description": "Consulta evidências sintéticas no DuckDB por SQL auditado.",
        "parameters": {"type": "object", "properties": {"report": {"type": "string", "enum": list(SQL)}},
                       "required": ["report"], "additionalProperties": False}, "strict": True}}
    messages = [
        {"role": "system", "content": "Você analisa dados sintéticos. Consulte a ferramenta antes de responder. Responda em português, cite números retornados, distinga associação de causa e alerta de fraude comprovada. A ferramenta oferece três relatórios fixos, sem filtros livres. Declare quando não conseguir responder à pergunta. Nunca invente dados."},
        {"role": "user", "content": question},
    ]
    evidence_list = []
    try:
        client = OpenAI(timeout=30, max_retries=1)
        for step in range(3):
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), messages=messages,
                tools=[tool], tool_choice="required" if step == 0 else "auto")
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return {"mode": "openai", "answer": message.content or "Sem resposta.", "evidence": evidence_list}
            for call in message.tool_calls:
                if call.function.name != "query_database":
                    raise ValueError("Ferramenta não permitida.")
                result = query_database(**json.loads(call.function.arguments))
                evidence_list.append(result)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, default=str, ensure_ascii=False)})
    except Exception as error:
        fallback["warning"] = f"API indisponível ({type(error).__name__}); usando fallback offline."
        return fallback
    fallback["warning"] = "Limite de rodadas atingido; usando fallback offline."
    return fallback


if __name__ == "__main__":
    print(json.dumps(explain("Explique o risco de churn"), ensure_ascii=False, default=str, indent=2))
