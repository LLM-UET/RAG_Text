from .BaseLLM import BaseLLM, Prompt
from typing import override, Any, Iterator

from .GeminiLLM import GeminiLLM

class LLM(BaseLLM):
    @override
    def do_init(self, model_ref: str, model_type: str, model_name: str, **kwargs: dict[str, Any]) -> None:
        if (model_type == 'gemini'):
            self.llm = GeminiLLM(model=model_ref, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @override
    def do_call(self, prompt: Prompt) -> str:
        return self.llm.call(prompt)
    
    @override
    def do_call_stream(self, prompt: Prompt)-> Iterator[str | None]:
        return self.llm.call_tokenstream(prompt)


