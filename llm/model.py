import torch
from huggingface_hub import login
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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

# OLD  local
# login(token="")
#
# _llm = None

# def get_llm():
#     global _llm
#
#     bnb = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",  # Gemma recommended
#     )
#
#     if _llm is None:
#         print(">>> Loading tokenizer...")
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
#         print(">>> Tokenizer OK")
#
#         print(">>> Loading model...")
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_ID,
#             device_map="auto",  # auto GPU/CPU placement
#             quantization_config=bnb,
#             torch_dtype=torch.float16,
#         )
#         print(">>> Model OK")
#
#         _llm = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=512,
#             temperature=0.3,
#             do_sample=True
#         )
#
#         print(">>> LLM pipeline ready")
#
#     return _llm
