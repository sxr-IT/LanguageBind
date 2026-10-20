"""Small CPU demo for LanguageBind's bundled image/audio/video examples.

The upstream inference.py assumes an NVIDIA CUDA device and loads all five
non-text modalities. This keeps its original code intact while allowing a
smaller first reproduction on a CPU machine.
"""

import argparse
import os
from pathlib import Path

import torch

from languagebind import (
    LanguageBind,
    LanguageBindImageTokenizer,
    to_device,
    transform_dict,
)


MODEL_NAMES = {
    "image": "LanguageBind_Image",
    "audio": "LanguageBind_Audio_FT",
    "video": "LanguageBind_Video_FT",
}
TEXTS = [
    "Training a parakeet to climb up a ladder.",
    "A lion climbing a tree to catch a monkey.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=tuple(MODEL_NAMES),
        default=["image", "audio"],
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("LANGUAGEBIND_CACHE", "D:/codex/languagebind_models")),
    )
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    clip_type = {modality: MODEL_NAMES[modality] for modality in args.modalities}
    model = LanguageBind(clip_type=clip_type, use_temp=False, cache_dir=str(args.cache_dir))
    model = model.to(device).eval()
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        "LanguageBind/LanguageBind_Image", cache_dir=str(args.cache_dir)
    )

    inputs = {
        "language": to_device(
            tokenizer(TEXTS, max_length=77, padding="max_length", truncation=True, return_tensors="pt"),
            device,
        )
    }
    for modality in args.modalities:
        suffix = ".mp4" if modality == "video" else ".wav" if modality == "audio" else ".jpg"
        paths = [str(Path("assets") / modality / f"{index}{suffix}") for index in range(2)]
        inputs[modality] = to_device(
            transform_dict[modality](model.modality_config[modality])(paths), device
        )

    with torch.inference_mode():
        embeddings = model(inputs)

    for modality in args.modalities:
        scores = embeddings[modality] @ embeddings["language"].T
        print(f"{modality} -> text cosine similarity:")
        print(scores.numpy())
        print("best text index per sample:", scores.argmax(dim=1).tolist())

    if "image" in embeddings and "audio" in embeddings:
        print("image -> audio cosine similarity:")
        print((embeddings["image"] @ embeddings["audio"].T).numpy())


if __name__ == "__main__":
    main()
