# Config file
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

data = {
    "NLI": {
        "dataset": "NLI",
        "DEFAULT_LN": "French",
        "dataset_type": "train",
        "batch_size": 32,
    },
    "MARC": {
        "dataset": "MARC",
        "DEFAULT_LN": "English",
        "dataset_type": "train",
        "batch_size": 32,
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

task = {
    "DEFAULT_TASK": "SA",
    "SUPPORTED_TASKS": ["SA", "NLI"],
}
