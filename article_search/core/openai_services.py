import os
import tiktoken
import logging
import dotenv
import openai
from dotenv import load_dotenv

from openai import OpenAI, AsyncOpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OpenAI.api_key = os.getenv("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def _truncate_text(self, text: str, is_embed: int, model: str) -> str:
        """Truncate text to fit within max token length"""
        encoding = tiktoken.encoding_for_model(model)

        embed_max_length = 250 # Embedding models should be shorter for batch embedding
        normal_max_length = 20000 # for text generation models
        if is_embed:
            max_length = embed_max_length
        else:
            max_length = normal_max_length
        
        tokens = encoding.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return encoding.decode(tokens)

    def generate_texts(self, model: str, prompt: str, query: str, max_output_tokens: int):
        """Generate text using OpenAI API"""
        is_embed = False
        truncated_query = self._truncate_text(query, is_embed, model)

        resp = self.client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompt
                    },
                {
                    "role": "user",
                    "content": truncated_query
                    }
            ],
            max_output_tokens=max_output_tokens,
        )
        return resp.output_text
    
    def async_generate_texts(self, model: str, prompt: str, query: str, max_output_tokens: int):
        """Generate text using Async OpenAI API"""
        is_embed = False
        truncated_query = self._truncate_text(query, is_embed, model)

        resp = self.async_client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompt
                    },
                {
                    "role": "user",
                    "content": truncated_query
                    }
            ],
            max_output_tokens=max_output_tokens,
        )
        return resp.output_text
    
    def embedder(self, model: str, texts: list):
        """Get embeddings using OpenAI API"""
        is_embed = True
        truncated_texts = self._truncate_text(texts, is_embed, model)

        model = "text-embedding-3-small"
        try:
            resp = self.client.embeddings.create(
                model=model,
                input=truncated_texts
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"Error during embedding: {e}")
            raise
       
    
    def batch_embedder(self, model: str, texts: list):
        """Get batch embeddings using OpenAI API"""
        is_embed = True
        truncated_texts = [self._truncate_text(text, is_embed, model) for text in texts]
        if model != "text-embedding-3-small":
            logger.info(f"changed model from {model} to text-embedding-3-small for batch embedding")
            model = "text-embedding-3-small"
    
        try:
            resp = self.client.embeddings.create(
                model=model,
                input=truncated_texts
                )

            # Extract the list of embedding vectors
            vectors = [item.embedding for item in resp.data]

            return vectors

        except Exception as e:
            print(f"Error during batch embedding: {e}")
            raise
        
