import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BlipProcessor, BlipForConditionalGeneration
from config import HF_MODEL, HF_TOKEN_INLINE, LLAMA_MODEL_PATH

logger = logging.getLogger(__name__)

class LLMManager:
    """Manage HF and Llama LLM models."""
    def __init__(self):
        self.hf_pipeline = None
        self.llama = None
        self.image_captioner = None  # Add this
        self._init_hf_model()
        self._init_llama()
        self._init_image_captioner()  # Add this call

    def _init_hf_model(self):
        """Load Hugging Face seq2seq model."""
        try:
            token_arg = {"use_auth_token": HF_TOKEN_INLINE} if HF_TOKEN_INLINE.strip() else {}
            tokenizer_hf = AutoTokenizer.from_pretrained(HF_MODEL, **(token_arg or {}))
            model_hf = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, device_map="auto", **(token_arg or {}))
            self.hf_pipeline = pipeline("text2text-generation", model=model_hf, tokenizer=tokenizer_hf)
            logger.info(f"Loaded HF model: {HF_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load HF model {HF_MODEL}: {e}")
            self.hf_pipeline = None

    def _init_llama(self):
        """Load Llama model (optional)."""
        try:
            from llama_cpp import Llama
            if Path(LLAMA_MODEL_PATH).exists():
                self.llama = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048)
                logger.info(f"Loaded Llama model from {LLAMA_MODEL_PATH}")
            else:
                logger.warning(f"Llama model not found at {LLAMA_MODEL_PATH}")
                self.llama = None
        except Exception as e:
            logger.warning(f"Llama not available: {e}")
            self.llama = None

    def _init_image_captioner(self):
        """Load BLIP image captioning model (lightweight)."""
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", device_map="auto")
            self.image_captioner = {"processor": processor, "model": model}
            logger.info("Loaded BLIP image captioning model")
        except Exception as e:
            logger.warning(f"Image captioner not available: {e}")
            self.image_captioner = None

    def answer(self, question, contexts):
        """Generate answer using available LLM."""
        if self.hf_pipeline:
            return self._answer_with_hf(question, contexts)
        elif self.llama:
            return self._answer_with_llama(question, contexts)
        else:
            raise RuntimeError("No LLM available")

    def _answer_with_hf(self, question, contexts, max_length=256, temperature=0.1):
        """Answer using Hugging Face model."""
        ctx = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = f"Use the sources to answer the question. If not present, say you don't know.\n\n{ctx}\n\nQuestion: {question}\nAnswer concisely:"
        out = self.hf_pipeline(prompt, max_length=max_length, temperature=temperature, do_sample=False)
        return out[0].get("generated_text") if out else ""

    def _answer_with_llama(self, question, contexts, max_tokens=512, temp=0.1):
        """Answer using Llama model."""
        ctx_text = "\n\n---\n\n".join(f"Source {i+1}:\n{c}" for i, c in enumerate(contexts))
        prompt = (
            "You are an assistant. Use the provided sources to answer the user's question. "
            "If the answer is not in the sources, say you don't know.\n\n"
            f"Sources:\n{ctx_text}\n\nQuestion: {question}\nAnswer concisely:"
        )
        resp = self.llama.create(prompt=prompt, max_tokens=max_tokens, temperature=temp)
        return resp["choices"][0]["text"].strip() if "choices" in resp and resp["choices"] else ""

    def caption_image(self, image_path):
        """Generate caption for an image."""
        if not self.image_captioner:
            return f"Image file: {Path(image_path).name} (captioning not available)"
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            processor = self.image_captioner["processor"]
            model = self.image_captioner["model"]
            
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            return caption
        except Exception as e:
            logger.warning(f"Image captioning failed: {e}")
            return f"Image file: {Path(image_path).name} (caption generation failed)"