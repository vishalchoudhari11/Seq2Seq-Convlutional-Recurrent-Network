import torch

# General sequence configuration
SAMPLE_RATE = 16000
DURATION_S = 5
BATCH_SIZE = 32

# STFT / frame parameters
WINDOW_LEN_MS = 16
HOP_LEN_MS = 4
OWS_MS = 8

WINDOW_LEN = int(WINDOW_LEN_MS / 1000 * SAMPLE_RATE)
HOP_LEN = int(HOP_LEN_MS / 1000 * SAMPLE_RATE)
OWS_LEN = int(OWS_MS / 1000 * SAMPLE_RATE)

ANALYSIS_WINDOW = torch.ones(WINDOW_LEN)

# SNR range (if generating synthetic noise)
SNR_RANGE = list(range(-10, 6))

# Paths (example)
DATASET_DIR = "./data"
SAVE_DIR = "./checkpoints"
