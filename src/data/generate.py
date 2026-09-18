"""Gera 24 meses sintéticos com sazonalidade, abandono e fraudes rotuladas."""
import duckdb
import holidays
import numpy as np
import pandas as pd

from src.config import DATA, SEED, prepare


def generate(n_accounts: int = 5000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula atividade diária; rótulos de fraude nunca entram nos preditores."""
    prepare()
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31")
    national_holidays = holidays.Brazil(years=[2024, 2025])
    opened = rng.integers(0, 300, n_accounts)
    abandoned = rng.random(n_accounts) < 0.38
    stop = np.where(abandoned, rng.integers(360, 760, n_accounts), 900)
    intensity = rng.uniform(0.25, 1.15, n_accounts)
    accounts = pd.DataFrame({
        "account_id": np.arange(n_accounts),
        "opened_at": dates[opened],
        "segment": rng.choice(["essencial", "premium"], n_accounts, p=[0.75, 0.25]),
    })
    chunks = []
    for day_index, date in enumerate(dates):
        weekday_factor = 0.72 if date.dayofweek >= 5 else 1.1
        monthly_factor = 1 + 0.20 * np.cos(2 * np.pi * (date.day - 5) / 30.44)
        holiday_factor = 0.60 if date in national_holidays else 1.0
        decline = np.clip((stop - day_index) / 65, 0.08, 1.0)
        rate = intensity * weekday_factor * monthly_factor * holiday_factor
        rate *= (1 + day_index / 1800) * decline
        rate *= (opened <= day_index) & (stop > day_index)
        counts = rng.poisson(rate)
        ids = np.repeat(np.arange(n_accounts), counts)
        size = len(ids)
        fraud = rng.random(size) < 0.003
        amount = rng.lognormal(4.1, 0.85, size)
        amount[fraud] *= rng.uniform(4, 18, fraud.sum())
        hour = rng.integers(7, 23, size)
        hour[fraud] = rng.integers(0, 24, fraud.sum())
        chunks.append(pd.DataFrame({
            "account_id": ids, "date": date, "amount": amount.round(2),
            "category": rng.choice(["PIX", "cartao", "boleto", "saque"], size,
                                   p=[0.46, 0.34, 0.15, 0.05]),
            "hour": hour, "is_fraud": fraud,
            "is_holiday": date in national_holidays,
        }))
    transactions = pd.concat(chunks, ignore_index=True)
    transactions.insert(0, "transaction_id", np.arange(len(transactions)))
    accounts.to_parquet(DATA / "accounts.parquet", index=False)
    transactions.to_parquet(DATA / "transactions.parquet", index=False)
    with duckdb.connect(str(DATA / "analytics.duckdb")) as connection:
        connection.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM accounts")
        connection.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM transactions")
    return accounts, transactions


if __name__ == "__main__":
    generate()
