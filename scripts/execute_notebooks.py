"""Valida todos os notebooks com kernels isolados e diretório explícito."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    """Executa e salva saídas para leitura direta no GitHub."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        print(f"Executando {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(root)}}).execute()
        nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
