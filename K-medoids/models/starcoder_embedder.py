
from typing import List, Dict, Any, Optional
import numpy as np

# Try to import transformers; if unavailable, we fallback to hashing
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False
    torch = None

from utils.hashing import hash_text

class StarCoderEmbedder:
    """
    Encodes suspect lists (code-like text) into a single vector using StarCoder.
    Fallback: hashing-based embedding if transformers/model aren't available.
    """
    def __init__(self, model_name: str = "bigcode/starcoder2-3b", device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.ok = False

        if _TRANSFORMERS_OK:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.model.to(device)
                self.model.eval()
                self.ok = True
            except Exception as e:
                print(f"[StarCoderEmbedder] Warning: could not load HF model '{model_name}': {e}")
                self.ok = False
        else:
            print("[StarCoderEmbedder] transformers not available; using hash fallback.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns an array of shape (len(texts), D).
        If hashing fallback is used, D=256. If StarCoder is used, D=model hidden size.
        """
        if not texts:
            return np.zeros((0, 256), dtype=np.float32)

        if self.ok:
            with torch.no_grad():
                embs = []
                for t in texts:
                    toks = self.tokenizer(
                        t, truncation=True, max_length=self.max_length, return_tensors="pt"
                    )
                    toks = {k: v.to(self.device) for k,v in toks.items()}
                    out = self.model(**toks)
                    # mean-pool over tokens (excluding padding if possible)
                    last = out.last_hidden_state  # [1, T, H]
                    attention_mask = toks.get("attention_mask", None)
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1)  # [1, T, 1]
                        summed = (last * mask).sum(dim=1)    # [1, H]
                        counts = mask.sum(dim=1).clamp(min=1) # [1, 1]
                        vec = (summed / counts).squeeze(0)    # [H]
                    else:
                        vec = last.mean(dim=1).squeeze(0)
                    vec = vec.cpu().numpy().astype("float32")
                    # L2 normalize
                    n = np.linalg.norm(vec) + 1e-8
                    embs.append(vec / n)
                return np.vstack(embs)
        else:
            # hashing fallback
            embs = []
            for t in texts:
                v = hash_text(t, dim=256)
                embs.append(v.astype("float32"))
            return np.vstack(embs)

    def embed_suspect_list(self, items: List[Dict[str, Any]]) -> np.ndarray:
        """
        items: list of {signature, line, score}
        We build a single text string concatenating signatures and weights.
        Returns a single vector (D,).
        """
        parts = []
        for it in items:
            sig = it.get("signature", "")
            line = it.get("line", "")
            score = it.get("score", 0.0)
            parts.append(f"{sig}#{line}|{score}")
        text = "\n".join(parts)
        return self.embed_texts([text])[0]
