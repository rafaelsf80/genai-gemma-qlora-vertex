""" Sample code to train a Gemma 7B model with QLoRA
    Uploads final merged model to Vertex AI Model Registry
    Must run on a V100, T4, L4 or above
"""

import glob
import json
import logging
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

from google.cloud import storage
from google.cloud import aiplatform


print(f"Notebook runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch version : {torch.__version__}")
print(f"Transformers version : {transformers.__version__}")
#print(f"Datasets version : {datasets.__version__}")
output_directory = os.environ['AIP_MODEL_DIR']
print(f"AIP_MODEL_DIR: {output_directory}")


model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


os.environ["WANDB_DISABLED"] = "true"
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers
from trl import SFTTrainer

def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
logging.info("Training ....")
trainer.train()
#logging.info("Evaluating ....")
#metrics = trainer.evaluate()


# Test Inference
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Save tokenizer, metrics and model locally
logging.info("Saving model and tokenizer locally ....")
tokenizer.save_pretrained(f'model_tokenizer')
trainer.save_model(f'model_output')
#logging.info('Saving metrics...')
#with open(os.path.join(f'model_output', 'metrics.json'), 'w') as f:
#    json.dump(metrics, f, indent=2)


logging.info("Saving model and tokenizer to GCS ....")
logging.info(f'Exporting SavedModel to: {output_directory}')

# extract GCS bucket_name from AIP_MODEL_DIR, ex: argolis-vertex-europewest4
bucket_name = output_directory.split("/")[2] # without gs://

# extract GCS object_name from AIP_MODEL_DIR, ex: aiplatform-custom-training-2023-02-22-16:31:12.167/model/
object_name = "/".join(output_directory.split("/")[3:])

directory_path = "model_output" # local
# Upload model to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


directory_path = "model_tokenizer" # local
# Upload tokenizer to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

# Upload metrics to Vertex AI Experiments

# PROJECT_ID = "argolis-rafaelsanchez-ml-dev"
# LOCATION = "europe-west4"
# EXPERIMENT_NAME = "exp-deepspeed"
# RUN_NAME = "run-001"

# Error: google.api_core.exceptions.NotFound: 
# 404 Resource not found.; GetContext is unable to find context resource with name: projects/989788194604/locations/europe-west4/metadataStores/default/contexts/exp-deepspeed-run-001
# aiplatform.init(experiment=EXPERIMENT_NAME, project=PROJECT_ID, location=LOCATION)

# aiplatform.start_run(run=RUN_NAME, resume=True)

# aiplatform.log_params(metrics)
