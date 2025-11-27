from .BaseLLM import BaseLLM, Prompt
from typing import override, Any, Callable, Iterator
from google import genai # type: ignore
from google.genai import types # type: ignore

class GeminiLLM(BaseLLM):
    @override
    def do_init(self, model_ref: str, model_type: str, model_name: str, **kwargs: dict[str, Any]) -> None:
        if (model_type != "gemini"):
            raise ValueError(f"Wrong model type: {model_type}")
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("API key is required for Gemini LLM.")
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string ; got " + str(type(api_key)) + " instead.")

        self.config: types.GenerateContentConfigDict = {}
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise ValueError("max_tokens must be an integer ; got " + str(type(max_tokens)) + " instead.")
            if max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer.")
        self.config["max_output_tokens"] = max_tokens

        temperature = kwargs.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, float):
                raise ValueError("temperature must be a float ; got " + str(type(temperature)) + " instead.")
            if temperature < 0.0 or temperature > 1.0:
                raise ValueError("temperature must be between 0.0 and 1.0.")
        self.config["temperature"] = temperature

        self.client = genai.Client(api_key=api_key)

    @override
    def do_call(self, prompt: Prompt) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config,
        )
        return response.text or ""
    
    @override
    def do_call_stream(self, prompt: Prompt) -> Iterator[str | None]:
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
        )
        for chunk in response_stream:
            yield chunk.text
