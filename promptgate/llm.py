
import os
import time
from openai import OpenAI

# USD per million input/output tokens.
# Free-tier usage is recorded as $0, not as a paid API estimate.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
}

_client = None


def _get_client():
    """Create the API client once and reuse it."""
    global _client

    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            max_retries=5,
            timeout=60.0,
        )

    return _client


def complete(system_prompt, user_input, model, temperature=0.0, retries=3):
    """Send one prompt to the LLM and return its response and metrics."""

    last_error = None

    for attempt in range(retries):
        try:
            start_time = time.perf_counter()

            response = _get_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
            )

            latency = time.perf_counter() - start_time

            input_tokens = (
                response.usage.prompt_tokens if response.usage else 0
            )
            output_tokens = (
                response.usage.completion_tokens if response.usage else 0
            )

            input_price, output_price = PRICES.get(
                model, (0.0, 0.0)
            )

            estimated_cost = (
                input_tokens * input_price
                + output_tokens * output_price
            ) / 1_000_000

            return {
                "text": response.choices[0].message.content or "",
                "latency": latency,
                "cost": estimated_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }

        except Exception as error:
            last_error = str(error)

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {
        "text": "",
        "latency": 0.0,
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": last_error,
    }
