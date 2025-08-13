import json
import os
from config import Config
os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_VISIBLE_DEVICES

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed
)
from trl import DPOTrainer, DPOConfig
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
# from waitGPU import wait_for_gpu_memory
# wait_for_gpu_memory(gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])
from tqdm import tqdm



def hsic_loss(A1, A2, sigma=Config.HSIC_SIGMA):

    def rbf_kernel(X, sigma):
        pairwise_dist = torch.cdist(X, X) ** 2
        return torch.exp(-pairwise_dist / (2 * sigma**2 + 1e-5))
    
    A1, A2 = A1.float(), A2.float()
    K = rbf_kernel(A1, sigma)
    L = rbf_kernel(A2, sigma)
    n = A1.size(0)
    H = torch.eye(n, device=A1.device) - torch.ones((n, n), device=A1.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = torch.trace(Kc @ Lc) / (n * n)
    return hsic

def mutual_info_loss_hsic(model, adapter1="lora1", adapter2="lora2"):
    loss = 0.0
    for name, module in model.named_modules():
        # This is an approximate implementation for quick debugging and running; it works as well.
        # The full version requires computing the AB product and will be open-sourced later.
        if hasattr(module, 'lora_A') and isinstance(module.lora_A, torch.nn.ModuleDict):
            if adapter1 not in module.lora_A or adapter2 not in module.lora_A:
                continue
            A1 = module.lora_A[adapter1].weight
            A2 = module.lora_A[adapter2].weight
            loss += hsic_loss(A1, A2)
    return loss

def freeze_adapter_params(model, adapter_name):

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and isinstance(module.lora_A, torch.nn.ModuleDict):
            if adapter_name in module.lora_A:
                for param in module.lora_A[adapter_name].parameters():
                    param.requires_grad = False
                for param in module.lora_B[adapter_name].parameters():
                    param.requires_grad = False

class DPOTrainerWithMI(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
  
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        lambda_orth = Config.ALPHA
        hsic = mutual_info_loss_hsic(model, "lora1", "lora2")
        total_loss = loss + lambda_orth * hsic
        
        if return_outputs:
            return total_loss, outputs
        return total_loss

def load_datasets():

    train_dataset = load_dataset("json", 
                               data_files=Config.get_dataset_path(Config.TAG1, 'train'), 
                               split="train")
    eval_dataset = load_dataset("json", 
                              data_files=Config.get_dataset_path(Config.TAG1, 'val'), 
                              split="train")
    return train_dataset, eval_dataset

def setup_model_and_tokenizer():

    model_path = Config.get_model_path()
    

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    adapter1_path = Config.get_adapter1_path()
    model = PeftModel.from_pretrained(model, adapter1_path, adapter_name="lora1")
    print(f"loading {adapter1_path}")
    
    lora_config = LoraConfig(**Config.LORA_CONFIG)
    model.add_adapter("lora2", lora_config)
    model.set_adapter("lora2")
    model.print_trainable_parameters()
    print("lora loading...")
    
    return model, tokenizer

def train_model():

    train_dataset, eval_dataset = load_datasets()
    

    model, tokenizer = setup_model_and_tokenizer()
    
    outputs_dir = Config.get_outputs_dir()
    dpo_config = DPOConfig(
        output_dir=f"{outputs_dir}/checkpoints",
        **Config.DPO_CONFIG
    )
    
    freeze_adapter_params(model, "lora1")
    
    trainer = DPOTrainerWithMI(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    print("training start")
    trainer.train()
    
    model.save_pretrained(outputs_dir)
    print(f"saved to {outputs_dir}")
    
    return outputs_dir

def evaluate_model_with_reward(tag, help_w, harm_w, lora_path):
    print(f"{lora_path}, Testing tag: {tag}")
    print(f'-------helpful: {help_w}, harmless: {harm_w}-------')
    
    test_dataset = load_dataset("json", 
                              data_files=Config.get_dataset_path(tag, 'val'))['train']
    
    reward_model_name = Config.get_reward_model_path(tag)
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    def reward_fn(query, response):
        prompt = query + response
        inputs = rm_tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            reward = reward_model(**inputs).logits[0].item()
        return torch.tensor(reward)
    
    base_model_name = Config.get_model_path()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # exploration
    lora_config1 = LoraConfig.from_pretrained(f"{lora_path}/lora1")
    lora_config1.lora_alpha = lora_config1.lora_alpha * help_w
    model = PeftModel.from_pretrained(base_model, f"{lora_path}/lora1", config=lora_config1)
    merge_model = model.merge_and_unload()
    
    lora_config = LoraConfig.from_pretrained(f"{lora_path}/lora2")
    lora_config.lora_alpha = lora_config.lora_alpha * harm_w
    model = PeftModel.from_pretrained(merge_model, f"{lora_path}/lora2", config=lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()
    rewards = []
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            prompt = example['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            response = model.generate(**inputs, max_new_tokens=Config.MAX_NEW_TOKENS)
            response_text = tokenizer.decode(response[0], skip_special_tokens=True)
            reward = reward_fn(prompt, response_text)
            rewards.append(reward.item())
    
    average_reward = sum(rewards) / len(rewards)
    print(f"{tag} Average reward: {average_reward}")
    return average_reward

def evaluate_all_combinations(lora_path):
    pi = {}
    S = Config.EXPLORATION_SPACE
    
    for help_w in S:
        for harm_w in S:
            harm_score = evaluate_model_with_reward(
                tag="harm", help_w=help_w, harm_w=harm_w, lora_path=lora_path)
            helpful_score = evaluate_model_with_reward(
                tag="helpful", help_w=help_w, harm_w=harm_w, lora_path=lora_path)
            print(f'\nhelpful_weight: {help_w}, harmless_weight: {harm_w}')
            print(f"harmless: {harm_score}")
            print(f'helpful: {helpful_score}')
            
            pi.update({(help_w, harm_w): (helpful_score, harm_score)})
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(Config.RESULTS_DIR, Config.RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(pi, f)
        print(f'saved to {results_path}')

def main():
    set_seed(Config.RANDOM_SEED)
    
    lora_path = train_model()
    
    evaluate_all_combinations(lora_path)

if __name__ == "__main__":
    main()
