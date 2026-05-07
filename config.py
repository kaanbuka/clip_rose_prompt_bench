"""Central configuration for the CLIP rose classification experiment."""

# Rose classes: directory key -> human-readable label.
ROSE_CLASSES = {
    "rosa_canina": "Rosa canina (Dog Rose / Kuşburnu)",
    "rosa_damascena": "Rosa damascena (Damask Rose)",
    "rosa_multiflora": "Rosa multiflora (Multiflora Rose)",
}

ROSE_KEYS = list(ROSE_CLASSES.keys())
ROSE_LABELS = list(ROSE_CLASSES.values())

# CLIP backbone. Alternatives: "ViT-L/14", "RN50".
CLIP_MODEL = "ViT-B/32"

# I/O paths (relative to repo root).
IMAGE_DIR = "data/sample_images"
RESULTS_DIR = "results"

# Experiment settings.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42
