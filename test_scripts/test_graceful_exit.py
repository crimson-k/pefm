"""CPU smoke test; also runnable with torchrun --nproc_per_node=2."""

import os
from pathlib import Path
import signal
import sys
import tempfile

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pefm.models import ActionPredictor
from pefm.training.train import run_epoch, save_state
from pefm.utils.graceful_exit import GracefulExit


def main():
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True)
    cfg = OmegaConf.create({"device": "cpu", "seed": 42, "max_grad_norm": 1.,
                            "max_batches": 0, "horizon": 1, "teacher_sha256": "test"})
    batch = {"embed": torch.randn(2, 21, 6), "action_segments": torch.randn(2, 21, 4, 14)}
    for split in ("train", "val"):
        stop = GracefulExit(accelerator)
        model = ActionPredictor(6, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model, optimizer = accelerator.prepare(model, optimizer)
        before = accelerator.unwrap_model(model).initial.weight.detach().clone()

        def send_signal(*args):
            if accelerator.process_index == accelerator.num_processes - 1:
                os.kill(os.getpid(), signal.SIGUSR1)

        hook = model.register_forward_hook(send_signal)
        run_epoch(model, [batch] * 3, cfg, optimizer if split == "train" else None,
                  2, split, accelerator, stop)
        hook.remove()
        assert stop.requested and stop.batch == 1
        core = accelerator.unwrap_model(model)
        assert torch.equal(before, core.initial.weight) == (split == "val")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stop.save(output, model, optimizer, cfg, 2, split, save_state)
            if accelerator.is_main_process:
                state = torch.load(output / "latest.pt", weights_only=False)
                assert state["epoch"] == (2 if split == "train" else 3)
                assert state["interrupted"] and state["batches_completed"] == 1
                assert state["split"] == split
                assert state["format"] == "pefm-embedding-predictor-v1"
                assert bool(state["optimizer"]["state"]) == (split == "train")
                for key, value in core.state_dict().items():
                    assert torch.equal(state["model"][key], value)
                snapshot, = output.glob("interrupted_*.pt")
                assert os.path.samefile(snapshot, output / "latest.pt")
                assert not list(output.glob("*.tmp"))
    accelerator.end_training()
    print("graceful exit tests passed", flush=True)


if __name__ == "__main__":
    main()
