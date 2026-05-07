"""Core CLIP zero-shot classification engine.

Handles model loading, image/text encoding (per strategy plus ensemble
averaging), cosine-similarity scoring, and full-experiment orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import clip
import numpy as np
import torch
from PIL import Image

from config import CLIP_MODEL, IMAGE_EXTENSIONS, ROSE_KEYS
from prompt_strategies import all_strategies


class CLIPRoseClassifier:
    """Zero-shot rose classifier using OpenAI CLIP."""

    def __init__(self, model_name: str = CLIP_MODEL, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device} ...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("Model loaded.\n")

    # Text encoding -----------------------------------------------------------

    def encode_text_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Encode a list of prompts and return their mean L2-normalised embedding."""
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)            # (N, D)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.mean(dim=0)                              # (D,)

    def build_class_embeddings(
        self, strategy_prompts: dict[str, list[str]]
    ) -> torch.Tensor:
        """Stack per-class prompt embeddings into a (C, D) normalised tensor."""
        embeddings = []
        for key in ROSE_KEYS:
            embeddings.append(self.encode_text_prompts(strategy_prompts[key]))
        class_emb = torch.stack(embeddings)                        # (C, D)
        class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        return class_emb

    def build_ensemble_embeddings(self) -> torch.Tensor:
        """Average class embeddings across all strategies into one ensemble set."""
        all_embs = [
            self.build_class_embeddings(strat_prompts)
            for strat_prompts in all_strategies().values()
        ]
        stacked = torch.stack(all_embs)                            # (S, C, D)
        ensemble = stacked.mean(dim=0)                             # (C, D)
        ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
        return ensemble

    # Image encoding ----------------------------------------------------------

    def encode_image(self, image_path: str | Path) -> torch.Tensor:
        """Encode a single image into a normalised (D,) embedding."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0)                                      # (D,)

    # Scoring -----------------------------------------------------------------

    def score(
        self, image_emb: torch.Tensor, class_emb: torch.Tensor
    ) -> np.ndarray:
        """Return softmax probabilities over classes for a single image embedding.

        Args:
            image_emb: shape (D,).
            class_emb: shape (C, D).

        Returns:
            np.ndarray of shape (C,) summing to 1.
        """
        # 100.0 is the conventional CLIP temperature scaling factor.
        logits = (class_emb @ image_emb) * 100.0
        return logits.softmax(dim=0).cpu().numpy()

    # Full experiment ---------------------------------------------------------

    def run_experiment(self, image_dir: str | Path) -> dict:
        """Run every strategy against every image under ``image_dir``.

        Expected layout::

            image_dir/
              rosa_canina/
              rosa_damascena/
              rosa_multiflora/

        Returns a dict with keys:
            images       -- list of (true_class_key, image_path)
            strategies   -- list of strategy names (incl. "ensemble")
            scores       -- np.ndarray, shape (N_images, N_strategies, N_classes)
            predictions  -- np.ndarray, shape (N_images, N_strategies)
            true_labels  -- np.ndarray, shape (N_images,) of int class indices
        """
        image_dir = Path(image_dir)
        strategies = all_strategies()
        strategy_names = list(strategies.keys()) + ["ensemble"]

        # Pre-compute class embeddings per strategy.
        class_embeddings: dict[str, torch.Tensor] = {}
        for name, strat_prompts in strategies.items():
            print(f"Encoding text prompts - strategy: {name}")
            class_embeddings[name] = self.build_class_embeddings(strat_prompts)
        print("Encoding text prompts - strategy: ensemble")
        class_embeddings["ensemble"] = self.build_ensemble_embeddings()
        print()

        # Collect (true_class_key, path) records.
        image_records: list[tuple[str, Path]] = []
        for class_key in ROSE_KEYS:
            class_dir = image_dir / class_key
            if not class_dir.exists():
                print(f"  [WARN] Directory not found: {class_dir}")
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    image_records.append((class_key, p))

        if not image_records:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                "Expected subdirectories: " + ", ".join(ROSE_KEYS)
            )

        N = len(image_records)
        S = len(strategy_names)
        C = len(ROSE_KEYS)

        scores_arr = np.zeros((N, S, C), dtype=np.float32)
        true_labels = np.array(
            [ROSE_KEYS.index(rec[0]) for rec in image_records], dtype=int
        )

        print(f"Processing {N} images across {S} strategies...")
        for i, (class_key, img_path) in enumerate(image_records):
            print(f"  [{i+1}/{N}] {img_path.name} ({class_key})")
            img_emb = self.encode_image(img_path)
            for j, strat_name in enumerate(strategy_names):
                scores_arr[i, j, :] = self.score(img_emb, class_embeddings[strat_name])

        predictions = scores_arr.argmax(axis=2)  # (N, S)

        return {
            "images": image_records,
            "strategies": strategy_names,
            "scores": scores_arr,
            "predictions": predictions,
            "true_labels": true_labels,
        }
