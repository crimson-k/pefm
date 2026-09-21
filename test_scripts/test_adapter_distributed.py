"""Four-GPU smoke test for the Phase-B Adapter training loop."""

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pefm.training.train_adapter import run_adapter_epoch


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        value = torch.tensor(float(index + 1))
        return {
            "hidden": value.reshape(1, 1, 1),
            "dit_grid": (1, 1, 1),
            "valid_len": 1,
            "num_views": 1,
            "teacher_batch": {"target": (2 * value).reshape(1, 1, 1)},
        }


class TinyDistiller(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(1, 1, bias=False)

    def forward(self, hidden, teacher_batch, **kwargs):
        del kwargs
        prediction = self.adapter(hidden.float()).mean()
        target = teacher_batch["target"].float().mean()
        loss = (prediction - target).square()
        return {"adapter": {
            "total": loss,
            "correct_cosine": -loss.detach(),
            "shuffled_spatial_cosine": -loss.detach() - 1,
            "shuffled_time_view_cosine": -loss.detach() - 1,
            "next_prior_cosine": -loss.detach(),
        }}


def first_item(items):
    return items[0]


def main():
    accelerator = Accelerator()
    if accelerator.num_processes != 4:
        raise AssertionError(f"Expected four processes, got {accelerator.num_processes}")
    torch.manual_seed(0)
    model = TinyDistiller().to(accelerator.device)
    initial_weight = model.adapter.weight.detach().clone()
    optimizer = torch.optim.SGD(model.adapter.parameters(), lr=1e-2)
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False, collate_fn=first_item)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    metrics = run_adapter_epoch(
        model, loader, accelerator.device, optimizer=optimizer, accelerator=accelerator,
    )
    weight = accelerator.unwrap_model(model).adapter.weight.detach()
    gathered = accelerator.gather(weight.reshape(1))
    if not torch.isfinite(torch.tensor(list(metrics.values()))).all():
        raise AssertionError(f"Non-finite distributed metrics: {metrics}")
    if torch.equal(weight, initial_weight):
        raise AssertionError("Adapter weight did not update")
    torch.testing.assert_close(gathered, gathered[0].expand_as(gathered))
    accelerator.print({
        "world_size": accelerator.num_processes,
        "samples": len(TinyDataset()),
        "final_weight": float(gathered[0]),
        "total": metrics["total"],
    })
    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
