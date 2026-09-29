from abc import ABC, abstractmethod

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.vision import ProviderError, _is_retryable


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str, task_type: str) -> tuple[list[float], dict]:
        """Return (embedding, usage). usage has input_tokens/output_tokens.

        task_type: Gemini task hint, e.g. "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY".
        Raises ProviderError on terminal failure (bad request, wrong dimensionality).
        """


class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dim: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _embed(self, text: str, task_type: str):
        return self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
        )

    def embed(self, text: str, task_type: str) -> tuple[list[float], dict]:
        try:
            response = self._embed(text, task_type)
        except genai_errors.ClientError as e:
            raise ProviderError(f"Gemini rejected the request ({e.code}): {e}") from e

        values = (response.embeddings or [None])[0]
        values = values.values if values else None
        if values is None or len(values) != self.dim:
            got = len(values) if values else 0
            raise ProviderError(f"Expected {self.dim}-dim embedding, got {got}")

        # The Gemini Developer API doesn't return token usage for embeddings; estimate from length.
        input_tokens = max(1, len(text) // 4)
        return values, {"input_tokens": input_tokens, "output_tokens": 0}
