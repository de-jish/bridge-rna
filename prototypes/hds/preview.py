"""Local NASA visual refinement preview using the unchanged application layout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import app as production

app = production.build_app()

if __name__ == "__main__":
    print("NASA refinement: http://127.0.0.1:8065", flush=True)
    app.run(host="127.0.0.1", port=8065, debug=False, dev_tools_ui=False)
