from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from train import S2SMaskGIT

num_decoding_steps = 10
temperature = 1.0
top_p = 0.85
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "/home/nicolvisser/workspace/SimonSynth/wandb/run-20260310_005512-j86e0bin/files/best.pt"
segments_dir = "/mnt/newt/workspace/zerosyl/output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech"
acoustic_units_dir = "/mnt/newt/data/acoustic-units/hubert-discrete/LibriSpeech"
out_dir = "output/resynthesized-waveforms/LibriSpeech"


class SemanticUnitsDataset(Dataset):
    def __init__(
        self,
        segments_dir: str,
        semantic_pad_id: int,
        segments_pattern: str = "**/*.pt",
        max_seqlen: int = 4096,
    ):
        self.segments_dir = segments_dir

        self.segments_paths = sorted(list(Path(segments_dir).glob(segments_pattern)))
        assert len(self.segments_paths) > 0, "No segment files found"

        self.semantic_pad_id = semantic_pad_id
        self.max_seqlen = max_seqlen

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

        rel_path = segments_path.relative_to(self.segments_dir)

        return rel_path, semantic_units


maskgit = S2SMaskGIT.from_pretrained(checkpoint_path).to(device)
maskgit.eval()

acoustic = torch.hub.load(
    "bshall/acoustic-model:main", "hubert_discrete", trust_repo=True
).to(device)

hifigan = torch.hub.load(
    "bshall/hifigan:main", "hifigan_hubert_discrete", trust_repo=True
).to(device)


dataset = SemanticUnitsDataset(
    segments_dir=segments_dir,
    semantic_pad_id=maskgit.semantic_pad_idx,
    segments_pattern="dev-clean/**/*.pt",
)

for rel_path, semantic_units in tqdm(dataset):

    semantic_units = semantic_units.to(device)

    with torch.inference_mode():
        acoustic_units_predicted = maskgit.generate(
            semantic_units, num_decoding_steps, temperature, top_p, device
        )
        mel = acoustic.generate(acoustic_units_predicted.unsqueeze(0)).transpose(1, 2)
        target = hifigan(mel).squeeze(0).cpu()

    out_path = Path(out_dir) / rel_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(out_path, target, 16000)
