"""Detecção transacional e monitoramento agregado em avaliações separadas."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from statsmodels.tsa.seasonal import STL

from src.config import FIGURES, REPORTS, SEED


def robust_score(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Retorna desvios absolutos escalados pelo MAD, protegido contra escala zero."""
    return np.abs(values - center) / max(1.4826 * scale, 1e-8)


def run(transactions: pd.DataFrame) -> list[dict]:
    """Ajusta em 2024, define limiares em jan–set/2025 e avalia out–dez/2025."""
    features = pd.DataFrame({
        "log_amount": np.log1p(transactions.amount),
        "night": (transactions.hour < 7).astype(int),
        "weekday": transactions.date.dt.dayofweek,
    })
    train_mask = transactions.date < "2025-01-01"
    validation_mask = (transactions.date >= "2025-01-01") & (transactions.date < "2025-10-01")
    test_mask = transactions.date >= "2025-10-01"
    model = IsolationForest(n_estimators=100, max_samples=2048, random_state=SEED, n_jobs=-1)
    model.fit(features.loc[train_mask].sample(min(120000, train_mask.sum()), random_state=SEED))
    center = float(np.median(features.loc[train_mask, "log_amount"]))
    scale = float(np.median(np.abs(features.loc[train_mask, "log_amount"] - center)))
    labels = transactions.loc[test_mask, "is_fraud"].to_numpy()
    scores = {
        "Isolation Forest": (-model.score_samples(features.loc[validation_mask]),
                             -model.score_samples(features.loc[test_mask])),
        "Z-score robusto": (robust_score(features.loc[validation_mask, "log_amount"].values, center, scale),
                            robust_score(features.loc[test_mask, "log_amount"].values, center, scale)),
    }
    rows = []
    figure, axis = plt.subplots(figsize=(7, 4))
    cases = transactions.loc[test_mask].copy()
    for name, (validation, score) in scores.items():
        threshold = float(np.quantile(validation, 0.997))
        predicted = score >= threshold
        rows.append({"model": name, "precision": float(precision_score(labels, predicted, zero_division=0)),
                     "recall": float(recall_score(labels, predicted)),
                     "average_precision": float(average_precision_score(labels, score)),
                     "threshold": threshold, "alerts": int(predicted.sum())})
        precision, recall, _ = precision_recall_curve(labels, score)
        axis.plot(recall, precision, label=name)
        if name == "Isolation Forest":
            cases["anomaly_score"] = score
            cases["alert"] = predicted
    axis.axhline(labels.mean(), ls="--", color="gray", label="Prevalência")
    axis.set(xlabel="Recall", ylabel="Precisão", title="Fraudes sintéticas • teste temporal")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "anomaly_pr.png", dpi=150)
    plt.close(figure)
    cases.sort_values("anomaly_score", ascending=False).head(100).to_csv(REPORTS / "anomaly_cases.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORTS / "anomaly_metrics.csv", index=False)
    # STL atua no total monetário diário. Uma avaliação controlada separada
    # evita confundir fraude individual com uma anomalia em toda a operação.
    daily = transactions.groupby("date").amount.sum()
    train = daily.loc[:"2025-09-30"]
    fit = STL(np.log1p(train), period=7, robust=True).fit()
    residual_center = float(np.median(fit.resid))
    residual_scale = float(np.median(np.abs(fit.resid - residual_center)))
    seasonal = pd.Series(fit.seasonal, index=train.index).groupby(train.index.dayofweek).mean()
    level = float(np.median(fit.trend[-28:]))
    test = daily.loc["2025-10-01":].copy()
    event_labels = np.zeros(len(test), dtype=bool)
    event_labels[[10, 28, 49, 75]] = True
    test.iloc[np.flatnonzero(event_labels)] *= 2.5
    residual = np.log1p(test.values) - level - seasonal.reindex(test.index.dayofweek).values
    score = robust_score(residual, residual_center, residual_scale)
    alert = score >= 4
    stl_metrics = {"model": "STL diário (choques separados)",
                   "precision": float(precision_score(event_labels, alert, zero_division=0)),
                   "recall": float(recall_score(event_labels, alert)),
                   "average_precision": float(average_precision_score(event_labels, score)),
                   "threshold": 4, "alerts": int(alert.sum())}
    pd.DataFrame([stl_metrics]).to_csv(REPORTS / "stl_anomaly_metrics.csv", index=False)
    pd.DataFrame({"date": test.index, "amount": test.values, "score": score,
                  "is_injected_event": event_labels, "alert": alert}).to_csv(REPORTS / "stl_events.csv", index=False)
    return rows + [stl_metrics]
