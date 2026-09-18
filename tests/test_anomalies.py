"""Casos degenerados do escore robusto."""
import numpy as np

from src.anomalies import robust_score


def test_zero_mad_remains_finite() -> None:
    """Uma amostra constante não causa divisão por zero."""
    scores = robust_score(np.array([10, 10, 30]), center=10, scale=0)
    assert np.isfinite(scores).all()
    assert scores[0] == 0
    assert scores[2] > scores[0]
