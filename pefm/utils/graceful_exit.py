"""Save at a completed batch boundary when a training worker receives SIGUSR1."""

import os
import signal

import torch


class GracefulExit:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.requested = False
        self.local_requested = False
        self.batch = 0
        signal.signal(signal.SIGUSR1, self.request)
        print(f"[exit] worker PID={os.getpid()}: kill -USR1 {os.getpid()}", flush=True)

    def request(self, signum, frame):
        # Only set a flag here; saving and distributed collectives belong in the loop.
        self.local_requested = True

    def sync(self):
        flag = torch.tensor(int(self.local_requested or self.requested), device=self.accelerator.device)
        self.requested = bool(self.accelerator.reduce(flag, "sum").item())
        return self.requested

    def save(self, output, model, optimizer, cfg, epoch, split, save_state):
        if self.accelerator.is_main_process:
            # A partial training epoch is repeated on resume; validation needs no replay.
            completed = epoch if split == "train" else epoch + 1
            latest = output / "latest.pt"
            save_state(latest, self.accelerator.unwrap_model(model), optimizer, cfg, completed,
                       interrupted=True, interrupted_epoch=epoch, split=split,
                       batches_completed=self.batch)
            snapshot = output / f"interrupted_epoch_{epoch:04d}_{split}_batch_{self.batch:06d}.pt"
            if snapshot.exists():
                snapshot.unlink()
            os.link(latest, snapshot)
            print(f"[exit] Saved {snapshot} and latest.pt; stopping.", flush=True)
        self.accelerator.wait_for_everyone()
