from huggingface_hub import InferenceClient
from config import MODEL_ID, HF_TOKEN

_client = None


def get_llm():
    global _client

    if _client is None:
        _client = InferenceClient(
            model=MODEL_ID,
            api_key=HF_TOKEN
        )

    return _client