from unsloth import FastLanguageModel
import re
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/llm/Llama-3.2-1B-Instruct-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

attention_mask = inputs.ne(tokenizer.pad_token_id).int().to("cuda")

response = model.generate(
    input_ids = inputs,
    attention_mask=attention_mask,
    max_new_tokens = 16,
    use_cache = True, temperature = 1.5, min_p = 0.1
    )

generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

match = re.search(r"assistant\n(.*)", generated_text, flags=re.DOTALL)

if match:
    cleaned_response = match.group(1).strip()
else:
    cleaned_response = "No response found after 'assistant'."

print(cleaned_response)