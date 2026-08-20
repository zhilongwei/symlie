"""Execute every example notebook as a regression test."""

import json
from pathlib import Path

import pytest
from IPython.display import display

EXAMPLES = Path(__file__).parents[1] / "examples"
NOTEBOOKS = tuple(sorted(EXAMPLES.glob("*.ipynb")))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_code(notebook):
    """Run code cells in notebook order without modifying the notebook file."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )
    namespace = {"display": display, "__name__": "__main__"}

    # The notebooks are version-controlled project inputs, not external data.
    exec(compile(source, str(notebook), "exec"), namespace)  # noqa: S102
