import os, random, datetime, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p