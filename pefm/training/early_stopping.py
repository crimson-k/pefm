"""Track the true best model separately from significant loss improvements."""

import math


def initial_state(value, epoch):
    return {
        "best": float(value), "best_epoch": int(epoch),
        "significant_best": float(value), "significant_epoch": int(epoch),
        "stop": False,
    }


def update(state, value, epoch, check_every=10, patience=30, min_relative_improvement=0.005):
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}: {value}")
    improved = value < state["best"]
    if improved:
        state.update(best=float(value), best_epoch=int(epoch))
    if epoch % check_every == 0:
        if value <= state["significant_best"] * (1 - min_relative_improvement):
            state.update(significant_best=float(value), significant_epoch=int(epoch))
        state["stop"] = epoch - state["significant_epoch"] >= patience
    return improved
