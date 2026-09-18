"""Teste integrado do dashboard e consulta offline."""
import pytest
from streamlit.testing.v1 import AppTest

from src.config import DATA, REPORTS, ROOT


@pytest.mark.skipif(not (DATA / "analytics.duckdb").exists() or not (REPORTS / "summary.json").exists(), reason="Requer pipeline completo")
def test_dashboard_and_offline_agent() -> None:
    """Renderiza abas e aciona a ferramenta SQL pelo dashboard."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.button[0].click().run()
    assert not app.exception
    assert any("probabilidade prevista" in item.value for item in app.markdown)
