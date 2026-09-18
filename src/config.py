"""Caminhos e configuração compartilhados."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42


def prepare() -> None:
    """Cria as pastas utilizadas pelo pipeline."""
    for path in (DATA, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
