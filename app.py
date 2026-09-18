"""Dashboard executivo do portfólio de analytics bancário sintético."""
import json

import pandas as pd
import streamlit as st

from src.agent import explain
from src.config import FIGURES, REPORTS

st.set_page_config(page_title="Digital Account Analytics", page_icon="📊", layout="wide")
st.title("Digital Account Analytics")
st.caption("Laboratório de decisão • 5.000 contas • 24 meses • dados 100% sintéticos")
if not (REPORTS / "summary.json").exists():
    st.info("Execute `python -m src.pipeline` para gerar dados e resultados.")
    st.stop()
summary = json.loads((REPORTS / "summary.json").read_text(encoding="utf-8"))
st.warning("Demonstração com dados sintéticos. Previsões e impacto não representam um banco real.")
tabs = st.tabs(["Visão executiva", "Série temporal", "Anomalias", "Churn", "Agente SQL"])
with tabs[0]:
    columns = st.columns(4)
    columns[0].metric("Contas", f"{summary['accounts']:,}")
    columns[1].metric("Transações", f"{summary['transactions']:,}")
    columns[2].metric("Movimentação", f"R$ {summary['total_amount'] / 1e6:.1f} mi")
    columns[3].metric("Fraudes injetadas", f"{summary['fraud_rate']:.2%}")
    st.image(str(FIGURES / "executive.png"))
    st.subheader("Retenção por coorte de abertura")
    retention = pd.read_csv(REPORTS / "sql_retention.csv")
    st.dataframe(retention.pivot(index="cohort", columns="age_months", values="retention").style.format("{:.0%}"))
    st.caption("Contas ativas no mês / contas abertas na coorte. Células futuras não observadas ficam vazias.")
with tabs[1]:
    st.image(str(FIGURES / "forecast.png"))
    st.dataframe(pd.read_csv(REPORTS / "forecast_metrics.csv"), hide_index=True)
    st.caption("Três janelas expansivas; horizonte de 28 dias. MAPE em %, RMSE em transações/dia.")
    with st.expander("Decomposição STL"):
        st.image(str(FIGURES / "stl.png"))
with tabs[2]:
    st.image(str(FIGURES / "anomaly_pr.png"))
    st.dataframe(pd.read_csv(REPORTS / "anomaly_metrics.csv"), hide_index=True)
    st.subheader("Fila de revisão — 100 maiores scores do teste")
    cases = pd.read_csv(REPORTS / "anomaly_cases.csv")
    category = st.selectbox("Modalidade", ["Todas"] + sorted(cases.category.unique()))
    st.dataframe(cases if category == "Todas" else cases.loc[cases.category == category], hide_index=True)
    st.subheader("STL: choques diários controlados, alvo distinto")
    st.dataframe(pd.read_csv(REPORTS / "stl_anomaly_metrics.csv"), hide_index=True)
with tabs[3]:
    st.dataframe(pd.read_csv(REPORTS / "churn_metrics.csv"), hide_index=True)
    st.image(str(FIGURES / "churn_evaluation.png"))
    st.image(str(FIGURES / "shap.png"))
    threshold = st.slider("Limiar para revisão de risco", 0.0, 1.0, 0.20, 0.05)
    predictions = pd.read_csv(REPORTS / "churn_predictions.csv")
    st.dataframe(predictions.loc[predictions.churn_probability >= threshold].sort_values("churn_probability", ascending=False), hide_index=True)
    st.caption("Ranking do teste reservado. SHAP explica a árvore base, antes da calibração. Associação não implica causa.")
with tabs[4]:
    st.write("Consulte um resumo executivo, o ranking de anomalias ou o ranking de churn.")
    question = st.text_input("Pergunta", "Explique o risco de churn")
    use_api = st.checkbox("Usar OpenAI, se OPENAI_API_KEY estiver configurada", value=False)
    if st.button("Consultar evidências"):
        try:
            result = explain(question, use_api=use_api)
            st.caption(f"Modo: {result['mode']}")
            st.write(result["answer"])
            if result.get("warning"):
                st.info(result["warning"])
            with st.expander("SQL e evidências"):
                st.json(result["evidence"])
        except Exception as error:
            st.error(f"Gere o banco com python -m src.pipeline. Detalhe: {error}")
