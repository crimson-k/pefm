"""CPU smoke test; also runnable with torchrun --nproc_per_node=2."""

import ast
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
from pefm.utils.graceful_exit import GracefulExit


class TinyEvaluator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, batch):
        return self.layer(batch["x"])

    def compute_loss(self, output, *args):
        loss = output.square().mean()
        return {"total": loss}


def main():
    accelerator = Accelerator(cpu=True)
    cfg = OmegaConf.create({
        "device": "cpu", "seed": 42, "max_grad_norm": 1.0, "max_batches": 0,
        "model": {"kl_free": 1.0, "loss_scales": {"dyn": 1.0, "rep": 0.1}},
    })
    loader = [{"x": torch.ones(2, 1), "reset": torch.zeros(2, dtype=torch.bool)}] * 3
    for filename in ("train.py", "train_eval_action.py"):
        # Execute the actual loop/save functions without loading video/model dependencies.
        path = ROOT / "pefm/training" / filename
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("run_epoch", "save_state")]
        scope = {"torch": torch, "os": os, "OmegaConf": OmegaConf}
        exec(compile(tree, str(path), "exec"), scope)
        for split in ("train", "val"):
            stop = GracefulExit(accelerator)
            model = TinyEvaluator()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            model, optimizer = accelerator.prepare(model, optimizer)
            before = accelerator.unwrap_model(model).layer.weight.detach().clone()

            def send_signal(*args):
                # Only the last rank receives USR1, during the first forward.
                if accelerator.process_index == accelerator.num_processes - 1:
                    os.kill(os.getpid(), signal.SIGUSR1)

            hook = model.register_forward_hook(send_signal)
            scope["run_epoch"](model, loader, cfg, optimizer if split == "train" else None,
                               2, split, accelerator, stop)
            hook.remove()
            assert stop.requested and stop.batch == 1
            core = accelerator.unwrap_model(model)
            assert torch.equal(before, core.layer.weight) == (split == "val")
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                stop.save(output, model, optimizer, cfg, 2, split, scope["save_state"])
                if accelerator.is_main_process:
                    state = torch.load(output / "latest.pt", weights_only=False)
                    assert state["epoch"] == (2 if split == "train" else 3)
                    assert state["interrupted"] and state["batches_completed"] == 1
                    assert state["split"] == split
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
