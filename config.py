import os

class Config:
    CUDA_VISIBLE_DEVICES = "4"
 
    BACKBONE = 'llama2'  
    MODEL_PATH = "./llama-2-hf" 
    LLAMA3_MODEL_PATH = "../llama-3-Instruct"  
    ALPHA = 10  
    TAG0 = "helpful"
    TAG1 = "harm"
    DATASET_BASE_PATH = "RLHF/dataset/"
    TRAIN_DATASET_SUFFIX = "_train_formed.jsonl"
    VAL_DATASET_SUFFIX = "_val_formed.jsonl"
    ADAPTER1_PATH_HARM = "outputs/llama2-7b-dpo_lora_harm"
    ADAPTER1_PATH_HELPFUL = "outputs/llama2-7b-dpo_lora_helpful"
    

    @classmethod
    def get_outputs_dir(cls):
        return f'./outputs/llama2-7b-olora2_{cls.TAG0}_{cls.TAG1}{cls.ALPHA}'
    
  
    LORA_CONFIG = {
        'r': 64,
        'lora_alpha': 128,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'lora_dropout': 0.05,
        'bias': "none",
        'task_type': "CAUSAL_LM"
    }

    DPO_CONFIG = {
        'per_device_train_batch_size': 2,
        'num_train_epochs': 3,
        'evaluation_strategy': "steps",
        'eval_steps': 2000,
        'save_strategy': "steps",
        'save_steps': 2000,
        'save_total_limit': 25,
        'load_best_model_at_end': True,
        'report_to': "none"
    }
    
    MAX_NEW_TOKENS = 200
    REWARD_MODEL_BASE_PATH = "RLHF/gpt2-reward models/"
    REWARD_MODEL_SUFFIX = "-reward model"

    EXPLORATION_SPACE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]#S space

    HSIC_SIGMA = 1.0
    

    RESULTS_DIR = 'outputs/results/'
    RESULTS_FILE = 'mva.json'
    

    RANDOM_SEED = 60
    
    @classmethod
    def get_model_path(cls):
        return cls.MODEL_PATH if cls.BACKBONE == 'llama2' else cls.LLAMA3_MODEL_PATH
    
    @classmethod
    def get_adapter1_path(cls):
        return cls.ADAPTER1_PATH_HARM if cls.TAG0 == "harm" else cls.ADAPTER1_PATH_HELPFUL
    
    @classmethod
    def get_reward_model_path(cls, tag):
        return f"{cls.REWARD_MODEL_BASE_PATH}{tag}{cls.REWARD_MODEL_SUFFIX}"
    
    @classmethod
    def get_dataset_path(cls, tag, split_type):
        suffix_map = {
            'train': cls.TRAIN_DATASET_SUFFIX,
            'val': cls.VAL_DATASET_SUFFIX
        }
        return f"{cls.DATASET_BASE_PATH}{tag}{suffix_map[split_type]}"