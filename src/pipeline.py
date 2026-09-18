"""Executa o portfólio completo sem necessidade de chave de API."""
import json
import time

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import anomalies, churn, eda, forecast
from src.config import DATA, FIGURES, REPORTS, SEED, prepare
from src.data.generate import generate


def run() -> dict:
    """Gera dados, métricas, gráficos e tabelas consultáveis pelo agente."""
    prepare()
    started = time.perf_counter()
    print("[1/6] Gerando dados sintéticos...", flush=True)
    accounts, transactions = generate()
    print("[2/6] SQL e inferência...", flush=True)
    summary = {"seed": SEED, "accounts": len(accounts), "transactions": len(transactions),
               "fraud_rate": float(transactions.is_fraud.mean()),
               "total_amount": float(transactions.amount.sum()), "eda": eda.run()}
    print("[3/6] Backtesting de séries temporais...", flush=True)
    summary["forecast"] = forecast.run(transactions)
    print("[4/6] Anomalias...", flush=True)
    summary["anomalies"] = anomalies.run(transactions)
    print("[5/6] Churn e SHAP...", flush=True)
    summary["churn"] = churn.run(accounts, transactions)
    with duckdb.connect(str(DATA / "analytics.duckdb")) as connection:
        for name in ("anomaly_cases", "churn_predictions"):
            frame = pd.read_csv(REPORTS / f"{name}.csv")
            connection.register("report_frame", frame)
            connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM report_frame")
            connection.unregister("report_frame")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    monthly = pd.read_csv(REPORTS / "sql_mau.csv", parse_dates=["month"])
    axes[0].plot(monthly.month, monthly.mau, color="#087e8b", linewidth=2)
    axes[0].set(title="Contas ativas mensais", ylabel="MAU")
    axes[0].tick_params(axis="x", rotation=30)
    ticket = pd.read_csv(REPORTS / "sql_ticket.csv")
    axes[1].bar(ticket.category, ticket.average_ticket, color="#162b46")
    axes[1].set(title="Ticket médio por modalidade", ylabel="R$")
    figure.tight_layout()
    figure.savefig(FIGURES / "executive.png", dpi=150)
    plt.close(figure)
    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    (REPORTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[6/6] Concluído em {summary['runtime_seconds']}s. Resultados em reports/.", flush=True)
    return summary


if __name__ == "__main__":
    run()
