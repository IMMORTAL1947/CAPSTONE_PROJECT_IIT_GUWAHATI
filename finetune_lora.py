



import torch
import GPUtil
import os

GPUtil.showUtilization()

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU instead")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from huggingface_hub import notebook_login
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft import LoraConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch.utils.checkpoint
from peft import PeftModel
if "COLAB_GPU" in os.environ:
  from google.colab import output
  output.enable_custom_widget_manager()

if "COLAB_GPU" in os.environ:
  !huggingface-cli login
else:
  notebook_login()

base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

from google.colab import files
files.upload()

from datasets import Dataset

# Read the text file line by line
with open("customer_support_clean.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")  # Split on double newline


formatted_data = [{"text": line.strip()} for line in lines if line.strip()]

# Create Hugging Face dataset
dataset = Dataset.from_list(formatted_data)



tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)



torch.utils.checkpoint.use_reentrant = False
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=False,
    fp16=True,
    max_steps=1000,
    logging_steps=10,
    save_strategy="epoch",
    save_steps=500,
    logging_dir="./logs",
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False
trainer.train()

trainer.save_model("finetuned_customer_support_model")



# Load base model & tokenizer again
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Load PEFT adapter
finetuned_model = PeftModel.from_pretrained(base_model, "finetuned_customer_support_model")

# Generate prediction
instruction = "How do I cancel my order?"
eval_prompt = f"### Instruction:\n{instruction}\n\n### Response:"

inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

finetuned_model.eval()
with torch.no_grad():
    outputs = finetuned_model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))




