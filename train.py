import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import amp, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

import wandb

from model import S2SMaskGIT, S2SMaskGITConfig


@dataclass
class TrainConfig:
    # --- Wandb details ---
    entity: str
    project: str
    name: str

    # --- General training control ---
    device: str
    dtype: torch.dtype
    accumulation_steps: int
    grad_clip_max_norm: float
    batch_size: int
    num_workers: int

    # --- Data configuration ---
    train_segments_dir: str
    train_segments_pattern: str
    train_acoustic_units_dir: str
    train_acoustic_units_pattern: str

    valid_segments_dir: str
    valid_segments_pattern: str
    valid_acoustic_units_dir: str
    valid_acoustic_units_pattern: str

    max_seqlen: int

    # --- Optimizer / learning rate schedule ---
    lr_init: float
    lr_max: float
    lr_final: float
    n_linear_steps: int
    n_decay_steps: int
    betas: tuple[float, float]
    weight_decay: float
    eps: float


def sample_maskgit_mask(seqlen: int):
    r = torch.cos(torch.rand(size=()) * math.pi / 2)
    mask = torch.rand(size=(seqlen,)) < r
    return mask


class AlignedSemanticAndAcousticUnitsDataset(Dataset):
    def __init__(
        self,
        segments_dir: str,
        acoustic_units_dir: str,
        semantic_pad_id: int,
        acoustic_pad_id: int,
        segments_pattern: str = "**/*.pt",
        acoustic_units_pattern: str = "**/*.pt",
        max_seqlen: int = 4096,
        mask: bool = True,
    ):
        self.segments_paths = sorted(list(Path(segments_dir).glob(segments_pattern)))
        assert len(self.segments_paths) > 0, "No segment files found"

        self.acoustic_units_paths = sorted(
            list(Path(acoustic_units_dir).glob(acoustic_units_pattern))
        )
        assert len(self.acoustic_units_paths) > 0, "No acoustic-unit files found"

        assert len(self.segments_paths) == len(
            self.acoustic_units_paths
        ), "Mismatch in number of segment and acoustic-unit files"

        self.semantic_pad_id = semantic_pad_id
        self.acoustic_pad_id = acoustic_pad_id
        self.max_seqlen = max_seqlen
        self.mask = mask

    def __len__(self) -> int:
        return len(self.segments_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        segments_path = self.segments_paths[idx]
        segments = torch.load(segments_path).long()  # (num_segments, 3)
        assert segments.ndim == 2
        assert segments.size(1) == 3

        seg_starts, seg_ends, seg_ids = segments.T
        seg_lengths = seg_ends - seg_starts
        semantic_units = torch.repeat_interleave(seg_ids, seg_lengths)

        acoustic_units_path = self.acoustic_units_paths[idx]
        acoustic_units = torch.load(acoustic_units_path).long()

        seqlen = min(len(semantic_units), len(acoustic_units), self.max_seqlen)
        semantic_units = semantic_units[:seqlen]
        acoustic_units = acoustic_units[:seqlen]

        if self.mask:
            mask = sample_maskgit_mask(seqlen)
        else:
            mask = None

        return semantic_units, acoustic_units, mask

    def collate_fn(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_units, acoustic_units, mask = zip(*batch)
        seqlens = torch.tensor([len(su) for su in semantic_units]).long()

        indices = torch.arange(seqlens.max()).unsqueeze(0)
        src_key_padding_mask = indices >= seqlens.unsqueeze(1)

        semantic_units = pad_sequence(
            semantic_units, batch_first=True, padding_value=self.semantic_pad_id
        )
        acoustic_units = pad_sequence(
            acoustic_units, batch_first=True, padding_value=self.acoustic_pad_id
        )
        mask = pad_sequence(mask, batch_first=True, padding_value=False)
        return semantic_units, acoustic_units, mask, src_key_padding_mask


class LinearRampCosineDecayScheduler(LRScheduler):
    """
    Custom learning rate scheduler that increases linearly for n_linear_steps,
    then decays with cosine annealing for n_decay_steps,
    then stays at lr_final for the remaining steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_linear_steps (int): Number of steps for linear increase.
        n_decay_steps (int): Number of steps for cosine decay.
        lr_init (float, optional): Initial learning rate. Default is 0.
        lr_max (float, optional): Maximum learning rate. Default is 1e-5.
        lr_final (float, optional): Final learning rate. Default is 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_linear_steps: int,
        n_decay_steps: int,
        lr_init: float,
        lr_max: float,
        lr_final: float,
    ):
        self.n_linear_steps = n_linear_steps
        self.n_decay_steps = n_decay_steps

        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step <= self.n_linear_steps:
            lr = self.lr_init + (self.lr_max - self.lr_init) * current_step / (
                self.n_linear_steps
            )
        elif current_step <= self.n_linear_steps + self.n_decay_steps:
            lr = (
                0.5
                * math.cos(
                    (current_step - self.n_linear_steps)
                    / (self.n_decay_steps)
                    * math.pi
                )
                + 0.5
            ) * (self.lr_max - self.lr_final) + self.lr_final
        else:
            lr = self.lr_final
        return [lr for _ in self.base_lrs]


class Trainer:
    def __init__(self, model_cfg: S2SMaskGITConfig, train_cfg: TrainConfig):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.model = S2SMaskGIT(model_cfg).to(self.train_cfg.device)

        # initialize optimizer with training config lr and params
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr_init,
            betas=self.train_cfg.betas,
            weight_decay=self.train_cfg.weight_decay,
            eps=self.train_cfg.eps,
        )

        self.scheduler = LinearRampCosineDecayScheduler(
            optimizer=self.optimizer,
            n_linear_steps=self.train_cfg.n_linear_steps,
            n_decay_steps=self.train_cfg.n_decay_steps,
            lr_init=self.train_cfg.lr_init,
            lr_max=self.train_cfg.lr_max,
            lr_final=self.train_cfg.lr_final,
        )

        # Use GradScaler only for float16; bfloat16 uses autocast without scaler
        self.scaler = (
            torch.amp.GradScaler() if self.train_cfg.dtype == torch.float16 else None
        )

        train_dataset = AlignedSemanticAndAcousticUnitsDataset(
            segments_dir=train_cfg.train_segments_dir,
            acoustic_units_dir=train_cfg.train_acoustic_units_dir,
            semantic_pad_id=self.model.semantic_pad_idx,
            acoustic_pad_id=self.model.acoustic_pad_idx,
            segments_pattern=train_cfg.train_segments_pattern,
            acoustic_units_pattern=train_cfg.train_acoustic_units_pattern,
            max_seqlen=train_cfg.max_seqlen,
        )
        valid_dataset = AlignedSemanticAndAcousticUnitsDataset(
            segments_dir=train_cfg.valid_segments_dir,
            acoustic_units_dir=train_cfg.valid_acoustic_units_dir,
            semantic_pad_id=self.model.semantic_pad_idx,
            acoustic_pad_id=self.model.acoustic_pad_idx,
            segments_pattern=train_cfg.valid_segments_pattern,
            acoustic_units_pattern=train_cfg.valid_acoustic_units_pattern,
            max_seqlen=train_cfg.max_seqlen,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            collate_fn=valid_dataset.collate_fn,
        )

        self.run = wandb.init(
            entity=train_cfg.entity,
            project=train_cfg.project,
            name=train_cfg.name,
            config={
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__,
            },
        )

        self.current_step = 0
        self.current_global_step = 0
        self.current_epoch = 0
        self.best_loss = math.inf
        self.pbar = None

    def train_step(self, batch):
        semantic_units, acoustic_units, mask, src_key_padding_mask = batch
        semantic_units = semantic_units.to(self.train_cfg.device)  # (B,T)
        acoustic_units = acoustic_units.to(self.train_cfg.device)  # (B,T)
        mask = mask.to(self.train_cfg.device)  # (B,T)
        src_key_padding_mask = src_key_padding_mask.to(self.train_cfg.device)  # (B,T)

        masked_acoustic_units = acoustic_units.clone()
        masked_acoustic_units[mask] = self.model.acoustic_mask_idx

        logits = self.model(
            semantic_units, masked_acoustic_units, src_key_padding_mask
        )  # (B, T, output_vocab_size)

        # set unmasked tokens in the target to the pad index so loss ignores them
        targets = acoustic_units.clone()
        targets[~mask] = self.model.acoustic_pad_idx

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.model.acoustic_pad_idx,
        )
        return loss

    def valid_step(self, batch):
        semantic_units, acoustic_units, mask, src_key_padding_mask = batch
        semantic_units = semantic_units.to(self.train_cfg.device)  # (B,T)
        acoustic_units = acoustic_units.to(self.train_cfg.device)  # (B,T)
        mask = mask.to(self.train_cfg.device)  # (B,T)
        src_key_padding_mask = src_key_padding_mask.to(self.train_cfg.device)  # (B,T)

        masked_acoustic_units = acoustic_units.clone()
        masked_acoustic_units[mask] = self.model.acoustic_mask_idx

        logits = self.model(
            semantic_units, masked_acoustic_units, src_key_padding_mask
        )  # (B, T, output_vocab_size)

        targets = acoustic_units.clone()
        targets[~mask] = self.model.acoustic_pad_idx

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.model.acoustic_pad_idx,
        )
        preds = logits.argmax(dim=-1)

        # calculate accuracy ONLY on the masked tokens
        correct = (preds == acoustic_units) & mask
        # prevent division by zero if nothing was masked in the batch
        accuracy = correct.sum() / (mask.sum() + 1e-8)

        return loss, accuracy

    def train_epoch(self):
        """Yields step information for one epoch of training"""
        self.model.train()
        self.optimizer.zero_grad()
        for loader_idx, batch in enumerate(self.train_loader):

            with amp.autocast(
                device_type=self.train_cfg.device, dtype=self.train_cfg.dtype
            ):
                loss = self.train_step(batch)

                if self.train_cfg.accumulation_steps > 1:
                    loss = loss / self.train_cfg.accumulation_steps

            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()

            loss = loss * self.train_cfg.accumulation_steps

            self.current_step += 1

            # Step optimizer when we've accumulated enough gradients or at epoch end
            if (
                self.current_step % self.train_cfg.accumulation_steps == 0
                or loader_idx + 1 == len(self.train_loader)
            ):
                # unscale before clipping if using scaler
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.train_cfg.grad_clip_max_norm
                )

                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.current_global_step += 1
                self.pbar.update(1)
                yield loss.detach()
                self.model.train()

        self.current_epoch += 1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        accuracies = []
        for batch in tqdm(
            self.valid_loader, desc="Validating", position=1, leave=False
        ):
            loss, accuracy = self.valid_step(batch)
            losses.append(loss.item())
            accuracies.append(accuracy.item())
        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)

        if loss >= self.best_loss:
            print("Validation loss did not improve.")
        else:
            self.best_loss = loss

            checkpoint_path = Path(self.run.dir) / f"best.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "cfg": self.model.cfg.__dict__,
            }
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)

        return loss, accuracy

    def train(
        self,
        max_global_step: int,
        log_every_n_global_steps: int,
        validate_every_n_global_steps: int,
    ):
        """
        Training loop that runs until max epochs or steps is reached.
        Args:
            num_epochs: Maximum number of epochs to train for
            num_steps: Maximum number of steps to train for
            validate_every: Number of steps between validation runs
        """
        if self.current_global_step >= max_global_step:
            print(f"Already trained up to {self.current_global_step} steps")
            return

        self.pbar = tqdm(
            total=max_global_step, desc="Training", unit="step", position=0, leave=True
        )

        while True:
            for train_loss in self.train_epoch():
                log_data = {
                    "train/loss": train_loss,
                    "trainer/epoch": self.current_epoch,
                    "trainer/lr": self.scheduler.get_lr()[0],
                }

                log_flag = self.current_global_step % log_every_n_global_steps == 0

                if self.current_global_step % validate_every_n_global_steps == 0:
                    val_loss, val_accuracy = self.validate()
                    log_data["val/loss"] = val_loss
                    log_data["val/accuracy"] = val_accuracy
                    log_flag = True

                if log_flag:
                    self.run.log(data=log_data, step=self.current_global_step)
                    self.pbar.set_postfix(
                        {
                            "epoch": f"{self.current_epoch}",
                            "lr": f"{self.scheduler.get_lr()[0]:.1e}",
                            "train/loss": f"{train_loss:.4f}",
                            "val/loss": f"{self.best_loss:.4f}",
                        }
                    )
                    self.pbar.update(0)

                if self.current_global_step >= max_global_step:
                    print(f"Reached max steps ({max_global_step})")
                    checkpoint_path = (
                        Path(self.run.dir) / f"step-{self.current_global_step}.pt"
                    )
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "cfg": self.model.cfg.__dict__,
                    }
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    self.pbar.close()
                    self.run.finish()
                    return


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model_cfg = S2SMaskGITConfig(
        num_semantic_types=9116,
        num_acoustic_types=100,
        d_model=768,
        num_layers=12,
        dropout=0.1,
    )

    train_cfg = TrainConfig(
        entity="zerospeech",
        project="SimonSynth",
        name=f"SimonSynth-ZeroSylCollapsed-v040-k-9116-hubert-discrete-LibriSpeech-train-clean-100",
        device="cuda",
        dtype=torch.bfloat16,
        accumulation_steps=1,
        grad_clip_max_norm=1.0,
        batch_size=64,
        num_workers=12,
        train_segments_dir="/mnt/newt/workspace/zerosyl/output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech",
        train_segments_pattern="train-clean-100/**/*.pt",
        train_acoustic_units_dir="/mnt/newt/data/acoustic-units/hubert-discrete/LibriSpeech",
        train_acoustic_units_pattern="train-clean-100/**/*.pt",
        valid_segments_dir="/mnt/newt/workspace/zerosyl/output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech",
        valid_segments_pattern="dev-clean/**/*.pt",
        valid_acoustic_units_dir="/mnt/newt/data/acoustic-units/hubert-discrete/LibriSpeech",
        valid_acoustic_units_pattern="dev-clean/**/*.pt",
        max_seqlen=750,  # 15 seconds
        lr_init=0,
        lr_max=2e-4,
        lr_final=2e-5,
        n_linear_steps=200,
        n_decay_steps=10000 - 1000,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-8,
    )

    trainer = Trainer(model_cfg, train_cfg)

    trainer.train(
        max_global_step=10000,
        log_every_n_global_steps=5,
        validate_every_n_global_steps=100,
    )
