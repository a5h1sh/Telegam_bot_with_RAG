import logging
from config import LLM_PROVIDER, HF_MODEL, HF_TOKEN_INLINE, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

class LLMProvider:
    """Unified LLM provider supporting multiple backends."""
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.llm = None
        self._init_provider()

    def _init_provider(self):
        if self.provider == "ollama":
            try:
                self._init_ollama()
                logger.info("✅ Ollama provider initialized successfully.")
            except Exception as e:
                logger.error(f"Ollama init failed: {e}")
                logger.warning("⚠️ Falling back to Hugging Face provider.")
                self.provider = "huggingface"
                self._init_huggingface() # Attempt to use HF as a backup
        elif self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _init_huggingface(self):
        """Initialize HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            token_arg = {"use_auth_token": HF_TOKEN_INLINE} if HF_TOKEN_INLINE.strip() else {}
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, **(token_arg or {}))
            model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, device_map="auto", **(token_arg or {}))
            self.client = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            logger.info(f"Loaded HF model: {HF_MODEL}")
        except Exception as e:
            logger.error(f"HF init failed: {e}")
            raise

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import requests
            self.client = {"base_url": OLLAMA_BASE_URL, "model": OLLAMA_MODEL}
            # Test connection
            requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL}")
        except Exception as e:
            logger.error(f"Ollama init failed: {e}")
            raise

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            self.client = openai
            logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.error(f"OpenAI init failed: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate response based on provider."""
        if self.provider == "huggingface":
            return self._generate_hf(prompt, max_length)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, max_length)
        elif self.provider == "openai":
            return self._generate_openai(prompt, max_length)
        else:
            raise RuntimeError("No LLM provider initialized")

    def _generate_hf(self, prompt: str, max_length: int) -> str:
        """Generate using HuggingFace model."""
        try:
            out = self.client(prompt, max_length=max_length, do_sample=False)
            return out[0].get("generated_text", "")
        except Exception as e:
            logger.error(f"HF generation failed: {e}")
            return ""

    def _generate_ollama(self, prompt: str, max_length: int) -> str:
        """Generate using Ollama."""
        try:
            import requests
            response = requests.post(
                f"{self.client['base_url']}/api/generate",
                json={"model": self.client["model"], "prompt": prompt, "stream": False}
            )
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def _generate_openai(self, prompt: str, max_length: int) -> str:
        """Generate using OpenAI."""
        try:
            response = self.client.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return ""