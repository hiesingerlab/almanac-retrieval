import os
import logging
import openai
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class Prompt(Enum):
    QA = "Generate a comprehensive and complete answer (but no more than 80 words) for a given question (Q) solely based on the provided context (C). You must only use information from the provided context (C). Use an unbiased and journalistic tone. If you cannot find a relevant answer, write \"I could not find a relevant answer.\""
    CAL = "Create an executable python script to answer question (Q) using only the provided context (C). You must only use information from the provided context (C) and question (Q). Mark everything as False unless mentioned in the question (Q). Output your answer as an executable python script. Think step by step."

class OpenAIGeneration:

    def __init__(self, model_name: str = "text-davinci-003", temperature: float = 0.0, stream: bool = False, api_key: Optional[str] = None):
        """
        :param model_name: The name of the model to use.
        :param temperature: The temperature to use. Higher values will make the model more creative, lower values will make it more conservative.
        :param stream: Whether to stream the results.
        :param api_key: The API key to use. If not provided, will use the OPENAI_API_KEY environment variable.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key

    def generate(self, prompt: str, context: str, meta: str, promptStyle: Prompt = Prompt.QA):
        """
        Generate text using the OpenAI API.
        :param prompt: The prompt to use.
        """
        max_tokens = 1000 if promptStyle == Prompt.CAL else 250 # Calculations need more tokens as they generate code
        prompt = promptStyle.value + f"\nC:{context}" + f"\nQ: {prompt}" + (f"\nU: {meta}" if meta else "") 
        if self.stream:
            pass
        else:
            try:
                return openai.Completion.create(model=self.model_name, prompt=prompt, temperature=self.temperature, max_tokens=max_tokens, stream=False).choices[0].text
            except openai.error.APIError as e:
                logger.error(f"OpenAI API returned an API Error: {e}")
                return "Internal Server Error. Please try again later."
            except openai.error.APIConnectionError as e:
                logger.error(f"Failed to connect to OpenAI API: {e}")
                return "Internal Server Error. Please try again later."
            except openai.error.RateLimitError as e:
                logger.error(f"OpenAI API request exceeded rate limit: {e}")
                return "Internal Server Error. Please try again later."
        