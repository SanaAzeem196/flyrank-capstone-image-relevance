import io
from abc import ABC, abstractmethod

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.api.schemas import ImageMetadata, PostMetadata

VISION_PROMPT = """Analyze this image and return structured metadata.
subject: what is the main subject? (e.g., "red fox")
taxon: broad common name, lowercase, singular (e.g., "fox")
category: one of animal, vehicle, food, architecture, landscape, other
attributes: up to 5 visual characteristics
caption: short human-readable caption
confidence: 0.0-1.0 confidence in this classification
"""

POST_ANALYSIS_PROMPT = """Analyze this blog post and return structured metadata.
subject: what is the post mainly about? (e.g., "red fox behavior")
taxon: broad common name of the subject, lowercase, singular (e.g., "fox"); empty string "" if the post has no identifiable subject
category: one of animal, vehicle, food, architecture, landscape, other
confidence: 0.0-1.0 confidence in this classification

Title: {title}
Body: {body}
"""


class ProviderError(Exception):
    """Terminal (non-retryable) provider failure — caller should not retry."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.APIError) and (exc.code == 429 or exc.code >= 500)


class VisionProvider(ABC):
    @abstractmethod
    def tag_image(self, image_bytes: bytes, mime_type: str) -> tuple[ImageMetadata, dict]:
        """Return (metadata, usage). usage has input_tokens/output_tokens.

        Raises ProviderError on terminal failure (bad request, malformed schema).
        Raises the underlying genai_errors.APIError if retries are exhausted.
        """

    @abstractmethod
    def analyze_post(self, title: str, body: str) -> tuple[PostMetadata, dict]:
        """Return (metadata, usage) for a post's extracted subject/taxon/category.

        Same failure semantics as tag_image.
        """


class GeminiFlashVision(VisionProvider):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _generate(self, contents: list, response_schema: type):
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

    def _call(self, contents: list, response_schema: type):
        try:
            response = self._generate(contents, response_schema)
        except genai_errors.ClientError as e:
            raise ProviderError(f"Gemini rejected the request ({e.code}): {e}") from e

        parsed = response.parsed
        if parsed is None:
            # Schema validation failed inside the SDK; surface it as terminal, no retry.
            raise ProviderError(f"Response did not match {response_schema.__name__} schema: {response.text!r}")

        usage = response.usage_metadata
        return parsed, {
            "input_tokens": (usage.prompt_token_count if usage else 0) or 0,
            "output_tokens": (usage.candidates_token_count if usage else 0) or 0,
        }

    def tag_image(self, image_bytes: bytes, mime_type: str) -> tuple[ImageMetadata, dict]:
        image = PILImage.open(io.BytesIO(image_bytes))
        return self._call([VISION_PROMPT, image], ImageMetadata)

    def analyze_post(self, title: str, body: str) -> tuple[PostMetadata, dict]:
        prompt = POST_ANALYSIS_PROMPT.format(title=title, body=body)
        return self._call([prompt], PostMetadata)
