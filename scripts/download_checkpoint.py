"""
Download the official Any4D pretrained checkpoint from Hugging Face.

Example:
    python scripts/download_checkpoint.py --output-dir checkpoints
"""

import argparse
import os

DEFAULT_REPO_ID = "airlabshare/any4d-checkpoint"
DEFAULT_FILENAME = "any4d_4v_combined.pth"


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    return parser


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.output_dir,
    )
    print(f"Checkpoint downloaded to: {path}")


if __name__ == "__main__":
    main()
