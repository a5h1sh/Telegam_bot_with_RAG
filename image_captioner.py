import logging
from pathlib import Path
from typing import List, Dict, Optional
from config import DEFAULT_IMAGE_MODEL, IMAGE_MODELS
from PIL import Image
import re
import warnings
import torch

# Monkeypatch deprecated torch.cuda.amp.autocast -> torch.amp.autocast('cuda', ...)
try:
    if hasattr(torch, "amp") and hasattr(torch.cuda, "amp"):
        # preserve signature by forwarding args/kwargs
        original = torch.cuda.amp.autocast  # keep reference if needed
        def _forward_autocast(*args, **kwargs):
            return torch.amp.autocast("cuda", *args, **kwargs)
        torch.cuda.amp.autocast = _forward_autocast
except Exception:
    pass

# Optionally silence the specific FutureWarning from clip_interrogator
warnings.filterwarnings("ignore",
                        message=".*torch.cuda.amp.autocast\\(args...\\) is deprecated.*",
                        category=FutureWarning)

logger = logging.getLogger(__name__)

class ImageCaptioner:
    """
    Generate a short caption and keyword tags for an image.
    Tries BLIP2 first, then Clip Interrogator. Falls back to basic heuristic.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_choice = model_name or DEFAULT_IMAGE_MODEL
        self.backend = None
        self._init_model()

    def _init_model(self):
        # BLIP2
        if self.model_choice == "blip2" or (self.model_choice not in IMAGE_MODELS and "blip2" in IMAGE_MODELS):
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                # default HF model id (change if you host local weights)
                model_id = "Salesforce/blip2-flan-t5-small"  # lightweight choice; change to larger if available
                self.processor = Blip2Processor.from_pretrained(model_id)
                self.model = Blip2ForConditionalGeneration.from_pretrained(model_id)
                self.backend = "blip2"
                logger.info("ImageCaptioner loaded BLIP2")
                return
            except Exception as e:
                logger.warning(f"BLIP2 load failed: {e}")

        # Clip Interrogator
        if self.model_choice == "clip_interrogator" or "clip_interrogator" in IMAGE_MODELS:
            # choose device for clip-interrogator (use 'cpu' to avoid CUDA autocast)
            requested_device = getattr(self, "model_choice", None)
            device = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"

            # Clip Interrogator init (use device variable)
            try:
                from clip_interrogator import Config, Interrogator
                cfg = Config(clip_model_name="ViT-L-14/openai", device=device)
                self.interrogator = Interrogator(cfg)
                self.backend = "clip_interrogator"
                logger.info(f"ImageCaptioner loaded Clip Interrogator on {device}")
                return
            except Exception as e:
                logger.warning(f"Clip Interrogator load failed: {e}")

        # llava (fallback placeholder)
        if self.model_choice == "llava" or "llava" in IMAGE_MODELS:
            # LLAVA integration can be added here — for now fall back to BLIP2/heuristic
            logger.info("LLAVA selected but not configured; falling back to heuristics")
            self.backend = None

    def _simple_keywords(self, text: str, top_n: int = 3) -> List[str]:
        text = text.lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        stop = {"the","and","for","that","with","this","from","have","were","which","when","your","you","are","not","but","was","what","can","will","how","all","any","our","a","an","in","on","of","to","is"}
        freqs = {}
        for w in words:
            if w in stop:
                continue
            freqs[w] = freqs.get(w, 0) + 1
        sorted_k = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        tags = [k for k,_ in sorted_k[:top_n]]
        # fallback: pick first N distinct words
        if not tags:
            seen = []
            for w in words:
                if w not in seen and w not in stop:
                    seen.append(w)
                if len(seen) >= top_n:
                    break
            tags = seen
        return tags[:top_n]

    def generate_caption(self, image_path: Path) -> Optional[Dict]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return None

        # BLIP2 path
        if getattr(self, "backend", None) == "blip2":
            try:
                inputs = self.processor(images=img, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_new_tokens=32)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                caption = caption.strip()
                keywords = self._simple_keywords(caption, top_n=3)
                return {"caption": caption, "keywords": keywords}
            except Exception as e:
                logger.warning(f"BLIP2 generation failed: {e}")

        # Clip Interrogator path
        if getattr(self, "backend", None) == "clip_interrogator":
            try:
                result = self.interrogator.interrogate(img)  # returns caption-like string
                caption = (result or "").strip()
                keywords = self._simple_keywords(caption, top_n=3)
                return {"caption": caption, "keywords": keywords}
            except Exception as e:
                logger.warning(f"ClipInterrogator generation failed: {e}")

        # Fallback heuristic: basic caption from filename or size
        try:
            caption = f"Image ({image_path.name})"
            keywords = self._simple_keywords(image_path.stem.replace("_", " "), top_n=3)
            return {"caption": caption, "keywords": keywords}
        except Exception as e:
            logger.error(f"Fallback caption failed: {e}")
            return None