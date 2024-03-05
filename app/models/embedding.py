import os
import openai
import tiktoken
import numpy as np
from itertools import islice
from typing import Iterable, List, Optional, Optional

class OpenAIEmbedding:
    def __init__(self, model_name: str = "text-embedding-ada-002", encoding_name: str = "cl100k_base", context_length: int = 1000, average: bool = False, api_key: Optional[str] = None):
        """
        :param model_name: The name of the model to use.
        :param api_key: The API key to use. If not provided, will use the OPENAI_API_KEY environment variable.
        """
        self.model_name = model_name
        self.encoding_name = encoding_name
        self.api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
        self.context_length = context_length
        self.average = average

    def _batched(self, iterable: Iterable, n: int = 1000):
        """Batch data into tuples of length n. The last batch may be shorter."""
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while (batch := tuple(islice(it, n))):
            yield batch 

    def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a given text."""
        return openai.Embedding.create(input=text, model=self.model_name)["data"][0]["embedding"]

    def _chunked_tokens(self, text: str, chunk_length: int = 1000):
        """Chunk a text into chunks of tokens of length chunk_length."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        chunks_iterator = self._batched(tokens, chunk_length)
        yield from chunks_iterator

    def embed_text(self, text: str) -> List[float]:
        """Get the embedding for a given text. If the text is too long, it will be chunked into smaller pieces."""
        text = text.replace("\n", " ")
        chunk_embeddings = []
        chunks = []
        chunk_lens = []
        for chunk in self._chunked_tokens(text, chunk_length=self.context_length):
            chunks.append(chunk)
            chunk_embeddings.append(self._get_embedding(chunk))
            chunk_lens.append(len(chunk))

        if self.average:
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
            chunk_embeddings = chunk_embeddings.tolist()
        return chunks, chunk_embeddings
