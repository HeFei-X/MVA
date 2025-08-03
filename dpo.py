import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import DPOTrainer
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

tag='helpful'
backbone='llama2'
savedir="./outputs/"+backbone+"-dpo_lora_results_"+tag


print('start tuning',savedir)
train_dataset = load_dataset("json", data_files="RLHF/dataset/"+tag+"_train_formed.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="RLHF/dataset/"+tag+"_val_formed.jsonl", split="train")
model_path = "./Llama-2-7b-hf"


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  


from trl import DPOTrainer,DPOConfig

#training

dpo_config = DPOConfig(
    output_dir=savedir+"/checkpoint",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config, 
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
print(type(model))
print(hasattr(model, "generate"))

trainer.train()
model.save_pretrained(savedir)
print(f"saved to {savedir}")

