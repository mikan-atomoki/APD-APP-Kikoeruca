"""
Offline data preprocessing: Generate degraded audio + pseudo-labels.

This is the CPU-intensive step (~6-10 hours for 500k samples).
Run once, then train from the generated manifests.

Usage:
    python -m training.preprocess \
        --librispeech_root data/LibriSpeech \
        --demand_root data/DEMAND \
        --output_dir data/manifests \
        --n_train 500000 --n_val 50000 --n_test 50000
"""

import argparse
from pathlib import Path

from training.config import AudioConfig, APDLabelConfig, DegradationConfig
from training.manifest import generate_manifest, balance_manifest


def collect_audio_files(root, extensions=(".flac", ".wav", ".mp3")) -> list[str]:
    """Recursively collect audio files from a directory."""
    root = Path(root)
    files = []
    for ext in extensions:
        files.extend(str(p) for p in root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for APD training")
    parser.add_argument("--librispeech_root", type=str, required=True,
                        help="Path to LibriSpeech directory (train-clean-100/360)")
    parser.add_argument("--demand_root", type=str, default=None,
                        help="Path to DEMAND noise dataset")
    parser.add_argument("--dns_noise_root", type=str, default=None,
                        help="Path to DNS Challenge noise set")
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    parser.add_argument("--n_train", type=int, default=500_000)
    parser.add_argument("--n_val", type=int, default=50_000)
    parser.add_argument("--n_test", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=0,
                        help="Number of parallel workers (0 = all CPU cores)")
    parser.add_argument("--oversample", type=float, default=1.5,
                        help="Oversample factor for label balancing (1.0 = no balancing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    print("Collecting audio files...")
    ls_root = Path(args.librispeech_root)
    clean_files = collect_audio_files(ls_root)
    print(f"  Clean speech files: {len(clean_files)}")

    # Noise files
    noise_files = []
    if args.demand_root:
        noise_files.extend(collect_audio_files(Path(args.demand_root)))
    if args.dns_noise_root:
        noise_files.extend(collect_audio_files(Path(args.dns_noise_root)))
    print(f"  Noise files: {len(noise_files)}")

    # Speaker files for competing/babble (use a subset of LibriSpeech)
    speaker_files = clean_files  # reuse clean speech as interference
    print(f"  Speaker files (for interference): {len(speaker_files)}")

    if not clean_files:
        print("ERROR: No clean speech files found!")
        return

    if not noise_files:
        print("WARNING: No noise files found. Only 'none' and speaker-based maskers will be used.")
        # Use clean files as noise fallback for testing
        noise_files = clean_files[:100]

    audio_config = AudioConfig()
    degradation_config = DegradationConfig()
    label_config = APDLabelConfig()

    do_balance = args.oversample > 1.0

    # Generate splits
    for split, n_samples, seed_offset in [
        ("train.jsonl", args.n_train, 0),
        ("val.jsonl", args.n_val, 1000000),
        ("test.jsonl", args.n_test, 2000000),
    ]:
        gen_n = int(n_samples * args.oversample) if do_balance else n_samples

        print(f"\n{'='*60}")
        print(f"Generating {split} ({gen_n} samples" +
              (f", balancing to {n_samples})" if do_balance else ")"))
        print(f"{'='*60}")

        manifest_path = generate_manifest(
            clean_files=clean_files,
            noise_files=noise_files,
            speaker_files=speaker_files,
            output_dir=output_dir,
            manifest_name=split,
            n_samples=gen_n,
            audio_config=audio_config,
            degradation_config=degradation_config,
            label_config=label_config,
            seed=args.seed + seed_offset,
            n_workers=args.n_workers,
        )

        if do_balance:
            print(f"\nBalancing {split} to ~{n_samples} samples...")
            balance_manifest(manifest_path, n_samples, seed=args.seed + seed_offset)

    print("\nDone! Manifests saved to:", output_dir)


if __name__ == "__main__":
    main()
