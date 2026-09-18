"""Previsões diárias com janelas expansivas e horizonte de 28 dias."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import FIGURES, REPORTS, SEED


def daily_volume(transactions: pd.DataFrame) -> pd.Series:
    """Conta transações por dia, incluindo dias sem atividade."""
    return transactions.groupby("date").size().asfreq("D", fill_value=0).astype(float)


def run(transactions: pd.DataFrame) -> dict:
    """Compara três modelos em três cortes temporais sem acessar o futuro."""
    series = daily_volume(transactions)
    series.rename("volume").to_csv(REPORTS / "daily_volume.csv")
    decomposition = STL(series, period=7, robust=True).fit()
    figure = decomposition.plot()
    figure.set_size_inches(11, 7)
    figure.suptitle("Decomposição semanal STL • dados sintéticos", y=1.01)
    figure.tight_layout()
    figure.savefig(FIGURES / "stl.png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    rows, predictions = [], []
    for fold, cutoff in enumerate([len(series) - 84, len(series) - 56, len(series) - 28], 1):
        train, actual = series.iloc[:cutoff], series.iloc[cutoff:cutoff + 28]
        baseline = np.tile(train.iloc[-7:].to_numpy(), 4)
        sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = sarima.fit(disp=False, maxiter=100)
        prophet = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                          daily_seasonality=False, uncertainty_samples=0)
        prophet.add_seasonality(name="monthly", period=30.44, fourier_order=3)
        prophet.add_country_holidays(country_name="BR")
        prophet.fit(pd.DataFrame({"ds": train.index, "y": train.values}), seed=SEED)
        forecasts = {
            "Sazonal ingênuo": baseline,
            "SARIMA": fitted.forecast(28).to_numpy(),
            "Prophet": prophet.predict(pd.DataFrame({"ds": actual.index})).yhat.to_numpy(),
        }
        for name, raw in forecasts.items():
            predicted = np.maximum(raw, 0)
            rows.append({"model": name, "fold": fold,
                         "mape": float(np.mean(np.abs((actual.values - predicted) / np.maximum(actual.values, 1))) * 100),
                         "rmse": float(np.sqrt(np.mean((actual.values - predicted) ** 2))),
                         "converged": bool(fitted.mle_retvals["converged"]) if name == "SARIMA" else True})
            predictions.extend({"date": date, "model": name, "actual": truth,
                                "predicted": value, "fold": fold}
                               for date, truth, value in zip(actual.index, actual.values, predicted))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS / "forecast_folds.csv", index=False)
    comparison = metrics.groupby("model")[["mape", "rmse"]].mean().sort_values("rmse")
    comparison.to_csv(REPORTS / "forecast_metrics.csv")
    pd.DataFrame(predictions).to_csv(REPORTS / "forecast_predictions.csv", index=False)
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(series.iloc[-112:], color="#162b46", label="Observado")
    for name, group in pd.DataFrame(predictions).groupby("model"):
        axis.plot(group.date, group.predicted, label=name, alpha=0.8)
    axis.set(title="Volume diário • backtesting de 3 × 28 dias", ylabel="Transações")
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "forecast.png", dpi=150)
    plt.close(figure)
    return comparison.reset_index().to_dict("records")
