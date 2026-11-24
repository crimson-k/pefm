import pytest

from pefm.training.early_stopping import initial_state, update


def test_small_improvements_keep_true_best_but_do_not_reset_patience():
    state = initial_state(1.0, 150)
    for epoch, value in ((160, 0.999), (170, 0.998), (180, 0.997)):
        assert update(state, value, epoch)
    assert state["best_epoch"] == 180
    assert state["significant_epoch"] == 150
    assert state["stop"]


def test_accumulated_improvement_resets_patience():
    state = initial_state(1.0, 150)
    update(state, 0.997, 160)
    update(state, 0.994, 170)
    assert state["significant_epoch"] == 170
    update(state, 0.994, 190)
    assert not state["stop"]
    update(state, 0.994, 200)
    assert state["stop"]


def test_best_tracks_every_epoch_but_stop_checks_only_every_ten():
    state = initial_state(1.0, 150)
    assert update(state, 0.999, 179)
    assert not state["stop"]
    update(state, 1.001, 180)
    assert state["stop"] and state["best_epoch"] == 179
    with pytest.raises(FloatingPointError):
        update(state, float("nan"), 190)


def test_exact_threshold_counts_as_significant():
    state = initial_state(1.0, 150)
    update(state, 0.995, 180)
    assert state["significant_epoch"] == 180 and not state["stop"]
