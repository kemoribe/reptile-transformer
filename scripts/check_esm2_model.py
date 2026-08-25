"""Verify that an ESM-2 checkpoint loads through Hugging Face Transformers."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing import ESM2_DIM, ProteinFeatureExtractor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a local or remote Hugging Face ESM-2 model."
    )
    parser.add_argument(
        "--model_dir",
        default=os.environ.get("ESM2_MODEL", "facebook/esm2_t12_35M_UR50D"),
        help="Hugging Face model directory or model name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extractor = ProteinFeatureExtractor(
        model_path=args.model_dir,
        output_dim=ESM2_DIM,
        use_gpu=False,
    )
    embedding = extractor.extract("MKTFFVAVL")

    if embedding.shape != (ESM2_DIM,):
        raise RuntimeError(
            f"Unexpected embedding shape: {embedding.shape}; expected ({ESM2_DIM},)"
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError("The ESM-2 embedding contains NaN or infinity.")

    result = {
        "status": "ok",
        "loader": "transformers.AutoModel",
        "model_class": extractor.model.__class__.__name__,
        "model_reference": str(args.model_dir),
        "hidden_size": int(extractor.model.config.hidden_size),
        "embedding_shape": list(embedding.shape),
        "embedding_finite": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
