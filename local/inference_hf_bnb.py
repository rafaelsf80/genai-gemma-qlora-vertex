""" Basic inference of Gemma 2B on GPU using quantized versions (4bit/8bit) and BitsAndBytes
    IMPORTANT: You must accept Gemma license conditions on Hugging Face page
"""

# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# Config for 4bit
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))