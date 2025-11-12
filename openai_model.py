import re
import os
from typing import Literal, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# .env ファイルから環境変数をロード
load_dotenv()

class OpenAIBase:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("APIキーが見つかりません。環境変数 'OPENAI_API_KEY' を設定してください。")
        self.client = OpenAI(api_key=self.api_key)
        self.usage_in = 0
        self.usage_out = 0

    def create_openai_query(self, query: str):
        return [
            {"role": "user", "content": query},
        ]

    def get_response(
        self,
        query: str,
        temperature=0,
        max_tokens=15000,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
    ) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.create_openai_query(query=query),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        usage_in = response.usage.prompt_tokens
        usage_out = response.usage.completion_tokens
        return response.choices[0].message.content, usage_in, usage_out


class LLM:
    GPT_3_OPENAI = "gpt-3.5-turbo"
    GPT_4O_OPENAI = "gpt-4o"
    GPT_4O_MINI_OPENAI = "gpt-4o-mini"

    MODEL_MAP = {
        "openai": {
            "gpt3": GPT_3_OPENAI,
            "gpt4o": GPT_4O_OPENAI,
            "gpt-4o-mini": GPT_4O_MINI_OPENAI,
            "gpt4o-mini": GPT_4O_MINI_OPENAI,
        },
    }

    def __init__(
        self,
        base: Literal["openai"],
        use_model: Literal["gpt3", "gpt4o", "gpt-4o-mini", "gpt4o-mini"],
    ):
        self.base = base
        self.use_model = use_model
        self.model = self.choose_model()

    def choose_model(self):
        try:
            if self.base == "openai":
                return OpenAIBase(self.MODEL_MAP[self.base][self.use_model])
            else:
                raise ValueError(f"Invalid base: {self.base}")
        except KeyError:
            raise ValueError(f"Invalid model: {self.use_model}")

    def get_response(self, prompt: str, temperature=0, max_tokens=4000, top_p=1) -> Tuple[str, int, int]:
        response, in_usage, out_usage = self.model.get_response(
            query=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
            
        return response, in_usage, out_usage