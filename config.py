# Config file
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

from src.prompts.empty_prompt import EmptyPrompt
from src.prompts.nli_prompt import NLIPrompt
from src.prompts.sa_prompt import SAPrompt

data = {
    "NLI": {
        "dataset": "NLI",
        "DEFAULT_LN": "French",
        "dataset_type": "train",
        "batch_size": 32,
        "supported_tasks": ["NLI"]
    },
    "MARC": {
        "dataset": "MARC",
        "DEFAULT_LN": "English",
        "dataset_type": "train",
        "batch_size": 32,
        "supported_tasks": ["SA"]
    }
}

model = {
    "DEFAULT_MODEL": "xlm-roberta-base",    
    "SUPPORTED_MODELS": {
        "xlm-roberta-base": {
            "model_constructor": AutoModelForMaskedLM.from_pretrained,
            "model_name": "xlm-roberta-base",       # model name for huggingface
        },
        "bigscience/bloom-560m": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": "bigscience/bloom-560m",       # model name for huggingface
        }
    }
}

task_config = {
    "DEFAULT_TASK": "SA",
    "SUPPORTED_TASKS": {
        "SA": {
            "label_map": {'5': 'positive', '4': 'positive', '3': 'positive',
                     '2': 'negative', '1': 'negative',  '0': 'negative'},
            "prompt_class": SAPrompt
        }, 
        "NLI": {
            "label_map": {'': ''},
            "prompt_class": NLIPrompt
        },
        "Empty": {
            "label_map": {'': ' '},
            "prompt_class": EmptyPrompt
        }
    },
}