"""Proteções contra vazamento temporal e censura de rótulos."""
import pandas as pd
import pytest

from src.churn import FEATURES, build_features


def fixture_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria contas com atividade recente, abandono e conta já inativa."""
    accounts = pd.DataFrame({"account_id": [1, 2, 3], "opened_at": pd.to_datetime(["2025-01-01"] * 3)})
    transactions = pd.DataFrame({"account_id": [1, 2, 3, 2],
                                 "date": pd.to_datetime(["2025-11-30", "2025-11-20", "2025-09-01", "2025-12-30"]),
                                 "amount": [100.0, 50.0, 70.0, 80.0]})
    return accounts, transactions


def test_future_does_not_change_features() -> None:
    """Modificar valores futuros não muda os preditores do corte."""
    accounts, transactions = fixture_data()
    original = build_features(accounts, transactions)
    transactions.loc[transactions.date > "2025-11-30", "amount"] = 999999
    changed = build_features(accounts, transactions)
    pd.testing.assert_frame_equal(original[FEATURES], changed[FEATURES])
    assert original.loc[1, "churn"] == 1
    assert original.loc[2, "churn"] == 0
    assert 3 not in original.index
    assert "is_fraud" not in FEATURES


def test_incomplete_future_rejected() -> None:
    """Impede classificar dias ainda não observados como inatividade."""
    accounts, transactions = fixture_data()
    with pytest.raises(ValueError, match="incompleta"):
        build_features(accounts, transactions, cutoff="2025-12-15")
