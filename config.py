# config.py
import random, numpy as np, torch

# ── 하이퍼파라미터 ───────────────────────────────────────────
ENV_NAME          = "CartPole-v1"
GAMMA             = 0.99
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 32
MEMORY_CAPACITY   = 50_000
EPS_START         = 1.0
EPS_END           = 0.01
EPS_DECAY         = 1000
TARGET_UPDATE_EPS = 5
TOTAL_EPISODES    = 1200
SAVE_INTERVAL     = 10

# ── 공통 시드 (재현성) ───────────────────────────────────────
GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """파이썬·NumPy·PyTorch 난수 시드를 한 번에 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
