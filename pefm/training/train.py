"""Stage 2 Phase A entry point: fixed embeddings -> linear B / GRU / linear D.

Run as a module or directly; the implementation is shared with train_predictor.
Phase B uses train_adapter; Phase C is owned by BWM.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pefm.training.train_predictor import main, run_epoch, save_state


if __name__ == "__main__":
    main()
