"""Churn observacional: ausência de transações nos 30 dias após o corte."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA, FIGURES, REPORTS, SEED

FEATURES = ["count_recent", "count_previous", "activity_ratio", "recency", "mean_amount", "tenure"]


def build_features(accounts: pd.DataFrame, transactions: pd.DataFrame,
                   cutoff: str = "2025-11-30") -> pd.DataFrame:
    """Usa somente histórico até o corte; exige 30 dias completos de futuro."""
    date = pd.Timestamp(cutoff)
    if transactions.date.max() < date + pd.Timedelta(days=30):
        raise ValueError("Janela futura incompleta: rótulos censurados.")
    history = transactions.loc[transactions.date <= date]
    recent = history.loc[history.date > date - pd.Timedelta(days=30)]
    previous = history.loc[(history.date > date - pd.Timedelta(days=60)) &
                           (history.date <= date - pd.Timedelta(days=30))]
    # Apenas contas ativas em algum momento dos últimos 30 dias são elegíveis.
    frame = accounts.set_index("account_id").loc[sorted(recent.account_id.unique())].copy()
    frame["count_recent"] = recent.groupby("account_id").size()
    frame["count_previous"] = previous.groupby("account_id").size()
    frame["count_previous"] = frame.count_previous.fillna(0)
    frame["activity_ratio"] = (frame.count_recent + 1) / (frame.count_previous + 1)
    frame["recency"] = (date - history.groupby("account_id").date.max()).dt.days
    frame["mean_amount"] = recent.groupby("account_id").amount.mean()
    frame["tenure"] = (date - frame.opened_at).dt.days
    future = transactions.loc[(transactions.date > date) & (transactions.date <= date + pd.Timedelta(days=30))]
    frame["churn"] = (~frame.index.isin(future.account_id)).astype(int)
    return frame


def run(accounts: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """CV estratificada, calibração interna e teste reservado por conta."""
    frame = build_features(accounts, transactions)
    train_ids, test_ids = train_test_split(frame.index, test_size=0.25, random_state=SEED, stratify=frame.churn)
    train, test = frame.loc[train_ids], frame.loc[test_ids]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "LightGBM": LGBMClassifier(n_estimators=120, max_depth=3, num_leaves=7,
                                  min_child_samples=30, verbosity=-1, random_state=SEED, n_jobs=2),
    }
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    probabilities = {}
    for name, estimator in models.items():
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
        scores = cross_val_score(calibrated, train[FEATURES], train.churn, cv=cv, scoring="roc_auc")
        calibrated.fit(train[FEATURES], train.churn)
        probability = calibrated.predict_proba(test[FEATURES])[:, 1]
        probabilities[name] = probability
        rows.append({"model": name, "cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std()),
                     "test_auc": float(roc_auc_score(test.churn, probability)),
                     "ks": float(ks_2samp(probability[test.churn == 1], probability[test.churn == 0]).statistic),
                     "brier": float(brier_score_loss(test.churn, probability))})
        fpr, tpr, _ = roc_curve(test.churn, probability)
        axes[0].plot(fpr, tpr, label=name)
        fraction, mean_probability = calibration_curve(test.churn, probability, n_bins=8, strategy="quantile")
        axes[1].plot(mean_probability, fraction, "o-", label=name)
    for axis in axes:
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.legend(fontsize=8)
    axes[0].set(title="ROC • teste reservado", xlabel="Falsos positivos", ylabel="Verdadeiros positivos")
    axes[1].set(title="Calibração", xlabel="Probabilidade prevista", ylabel="Frequência observada")
    figure.tight_layout()
    figure.savefig(FIGURES / "churn_evaluation.png", dpi=150)
    plt.close(figure)
    # Seleção exclusivamente pela validação cruzada; teste não escolhe o modelo.
    winner = max(rows, key=lambda row: row["cv_auc_mean"])["model"]
    probability = probabilities[winner]
    output = test[FEATURES + ["churn"]].copy()
    output["churn_probability"] = probability
    output.to_csv(REPORTS / "churn_predictions.csv")
    output.to_parquet(DATA / "churn_predictions.parquet")
    # SHAP explica a árvore base, não a transformação de calibração.
    tree = models["LightGBM"].fit(train[FEATURES], train.churn)
    explanation = shap.TreeExplainer(tree)(test[FEATURES])
    values = explanation.values
    importance = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values()
    importance.rename("mean_abs_shap").to_csv(REPORTS / "shap_importance.csv")
    pd.DataFrame(values, index=test.index, columns=FEATURES).rename_axis("account_id").to_csv(REPORTS / "shap_local.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    importance.plot.barh(ax=axis, color="#087e8b")
    axis.set(title="Drivers de churn • LightGBM base", xlabel="SHAP absoluto médio (log-odds)")
    figure.tight_layout()
    figure.savefig(FIGURES / "shap.png", dpi=150)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(REPORTS / "churn_metrics.csv", index=False)
    selected = probability >= 0.20  # Política pré-fixada; não otimizada no teste.
    return {"models": rows, "selected_model": winner, "eligible_accounts": len(frame),
            "test_accounts": len(test), "test_churn_rate": float(test.churn.mean()),
            "targeted_accounts": int(selected.sum()),
            "target_precision": float(precision_score(test.churn, selected, zero_division=0)),
            "target_recall": float(recall_score(test.churn, selected)),
            "true_positive_targets": int(((test.churn == 1) & selected).sum())}
