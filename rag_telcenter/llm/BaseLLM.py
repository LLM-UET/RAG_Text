from ..common import log_debug
from typing import Iterator, Any
import pandas as pd

type Prompt = Any

class BaseLLM:
    def __init__(self, model: str, **kwargs: Any) -> None:
        log_debug(f"Initializing LLM model: {model}...")
        self.model_ref = model
        data = model.split("/")
        if len(data) != 2:
            raise ValueError(f"Invalid model reference: {model}")
        self.model_type = data[0]
        if self.model_type not in ["gemini"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.model_name = data[1]
        self.do_init(self.model_ref, self.model_type, self.model_name, **kwargs)
    
    def do_init(self, model_ref: str, model_type: str, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the model based on the model type and name.
        This method should be overridden by subclasses to provide
        specific initialization logic for different model types.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def do_call(self, prompt: Prompt) -> str:
        raise NotImplementedError
    
    def do_call_stream(self,
        prompt: Prompt,
    )-> Iterator[str | None]:
        raise NotImplementedError
    
    def call(self, prompt: Prompt) -> str:
        start_time = pd.Timestamp.now()
        try:
            return self.do_call(prompt)
        finally:
            end_time = pd.Timestamp.now()
            elapsed_time = (end_time - start_time).total_seconds()
            log_debug(f"LLM {self.model_ref} .call() took {elapsed_time:.2f} seconds")
    
    def call_tokenstream(self,
        prompt: Prompt,
    )-> Iterator[str | None]:
        """
        onTokenArrival: A callback function that will be called with each token
        as it arrives. The function should accept a single argument, which is the
        token string. The function can also accept None to indicate the end of the stream.
        """
        return self.do_call_stream(prompt)
