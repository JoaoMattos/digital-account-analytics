"""SQL puro, inferência por conta e estatística descritiva."""
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA, REPORTS

QUERIES = {
    "dau": "SELECT date, count(DISTINCT account_id) AS dau FROM transactions GROUP BY 1 ORDER BY 1",
    "mau": "SELECT date_trunc('month', date) AS month, count(DISTINCT account_id) AS mau FROM transactions GROUP BY 1 ORDER BY 1",
    "ticket": "SELECT category, count(*) AS transactions, avg(amount) AS average_ticket FROM transactions GROUP BY 1 ORDER BY 1",
    "retention": """
        WITH cohorts AS (
          SELECT account_id, date_trunc('month', opened_at) AS cohort FROM accounts
        ), activity AS (
          SELECT DISTINCT account_id, date_trunc('month', date) AS month FROM transactions
        ), sizes AS (SELECT cohort, count(*) AS cohort_size FROM cohorts GROUP BY 1)
        SELECT c.cohort, a.month, date_diff('month', c.cohort, a.month) AS age_months,
               count(*) AS active_accounts, max(s.cohort_size) AS cohort_size,
               count(*)::DOUBLE / max(s.cohort_size) AS retention
        FROM cohorts c JOIN activity a USING(account_id) JOIN sizes s USING(cohort)
        GROUP BY 1, 2 ORDER BY 1, 2
    """,
}


def run() -> dict:
    """Executa SQL e testes exploratórios sem tratar transações como independentes."""
    with duckdb.connect(str(DATA / "analytics.duckdb"), read_only=True) as connection:
        for name, query in QUERIES.items():
            connection.execute(query).df().to_csv(REPORTS / f"sql_{name}.csv", index=False)
        account_stats = connection.execute("""
            SELECT a.account_id, a.segment, avg(t.amount) AS average_ticket,
                   max(t.date) < DATE '2025-12-02' AS inactive
            FROM accounts a JOIN transactions t USING(account_id) GROUP BY 1, 2
        """).df()
    groups = [group.average_ticket.to_numpy() for _, group in account_stats.groupby("segment")]
    test = stats.ttest_ind(*groups, equal_var=False)
    chi = stats.chi2_contingency(pd.crosstab(account_stats.segment, account_stats.inactive))
    values = account_stats.average_ticket
    interval = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
    account_stats.describe().to_csv(REPORTS / "descriptive_statistics.csv")
    return {
        "account_mean_ticket": float(values.mean()), "ticket_ci95": list(interval),
        "welch_pvalue": float(test.pvalue), "chi2_pvalue": float(chi.pvalue),
        "inactive_share": float(np.mean(account_stats.inactive)),
    }
