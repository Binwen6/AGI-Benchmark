import litellm
import asyncio
import json
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel

litellm.drop_params = True  # Automatically drop unsupported parameters per provider

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    def __init__(self, model_name: str, use_system_fallback: bool = False):
        """
        model_name: The litellm-compatible string, e.g. 'openai/gpt-4o', 'anthropic/claude-3-haiku'.
        """
        self.model_name = model_name
        self.use_system_fallback = use_system_fallback

    async def async_prompt(self, prompt_text: str, system_instructions: str) -> str:
        """
        Issue one prompt call asynchronously.
        Handles system instruction fallback if the target API refuses system variables.
        """
        if self.use_system_fallback:
            combined_prompt = f"{system_instructions}\n\n{prompt_text}"
            messages = [{"role": "user", "content": combined_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt_text}
            ]

        for attempt in range(5):
            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                # Detection for non-supported system instructions
                if "Developer instruction" in err_str or "system" in err_str.lower() or "invalid_request_error" in err_str:
                    if not self.use_system_fallback:
                        self.use_system_fallback = True
                        combined_prompt = f"{system_instructions}\n\n{prompt_text}"
                        messages = [{"role": "user", "content": combined_prompt}]
                        continue # retry immediately
                
                if attempt == 4:
                    print(f"  [Error] Inference failed after 5 attempts on {self.model_name}: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)

    async def structured_prompt(self, prompt_text: str, system_instructions: str, schema: Type[T]) -> Optional[T]:
        """
        Perform a structured prompt. Relies on litellm passing Pydantic models to `response_format`.
        """
        if self.use_system_fallback:
            combined_prompt = f"{system_instructions}\n\n{prompt_text}"
            messages = [{"role": "user", "content": combined_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt_text}
            ]
        
        for attempt in range(5):
            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema
                )
                content = response.choices[0].message.content
                # If the API returned a string (some fallbacks might), try parsing it directly.
                # litellm returns parsed content in some cases if response_format is provided.
                if isinstance(content, str):
                    try:
                        return schema.model_validate_json(content)
                    except Exception:
                        pass
                
                # Some versions of litellm / models put it in a specific field, but Pydantic parse is robust
                return schema.model_validate_json(content)
            except Exception as e:
                if attempt == 4:
                    print(f"  [Error] Structured inference failed after 5 attempts on {self.model_name}: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
