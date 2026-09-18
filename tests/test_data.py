"""Contratos da simulação e persistência."""
import duckdb
import pandas as pd

from src.data import generate as generator


def test_generator_reproducible_and_persisted(tmp_path, monkeypatch) -> None:
    """Mesmo seed preserva os dados; chaves e rótulos mantêm integridade."""
    monkeypatch.setattr(generator, "DATA", tmp_path)
    accounts, transactions = generator.generate(n_accounts=80)
    again_accounts, again_transactions = generator.generate(n_accounts=80)
    pd.testing.assert_frame_equal(accounts, again_accounts)
    pd.testing.assert_frame_equal(transactions, again_transactions)
    assert transactions.transaction_id.is_unique
    assert transactions.account_id.isin(accounts.account_id).all()
    assert (transactions.amount > 0).all()
    assert 0.001 < transactions.is_fraud.mean() < 0.005
    assert transactions.date.max() == pd.Timestamp("2025-12-31")
    assert transactions.is_holiday.any()
    with duckdb.connect(str(tmp_path / "analytics.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM transactions").fetchone()[0] == len(transactions)
    assert (tmp_path / "accounts.parquet").exists()
