import torch
import torchaudio
from IPython.display import Audio, display
from zerosyl import ZeroSylCollapsed

# wget https://storage.googleapis.com/zerospeech-checkpoints/samples/1272-128104-0000.flac
waveform_path = "1272-128104-0000.flac"
num_decoding_steps = 10
temperature = 1.0
top_p = 0.85
device = "cuda" if torch.cuda.is_available() else "cpu"

wav, sr = torchaudio.load(waveform_path)

zerosyl = ZeroSylCollapsed.from_remote()

maskgit = torch.hub.load(
    "nicolvisser/S2SMaskGIT:maskter", "s2smaskgit", trust_repo=True
).to(device)

acoustic = torch.hub.load(
    "bshall/acoustic-model:main", "hubert_discrete", trust_repo=True
).to(device)

hifigan = torch.hub.load(
    "bshall/hifigan:main", "hifigan_hubert_discrete", trust_repo=True
).to(device)


with torch.inference_mode():

    seg_starts, seg_ends, seg_ids = zerosyl.encode(wav)
    sef_lengths = seg_ends - seg_starts
    semantic_units = torch.repeat_interleave(seg_ids, sef_lengths)

    acoustic_units = maskgit.generate(
        semantic_units, num_decoding_steps, temperature, top_p
    )

    mel = acoustic.generate(acoustic_units.unsqueeze(0)).transpose(1, 2)

    target = hifigan(mel).squeeze(0).cpu()

display(Audio(target, rate=16000))
