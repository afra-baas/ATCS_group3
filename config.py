# Config file
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from src.prompts.empty_prompt import EmptyPrompt
from src.prompts.nli_prompt import NLIPrompt
from src.prompts.sa_prompt import SAPrompt

data = {
    "NLI": {
        "dataset": "NLI",
        "DEFAULT_LN": "English",
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
    "DEFAULT_MODEL": "bloom",
    "SUPPORTED_MODELS": {
        "roberta": {
            "model_constructor": AutoModelForMaskedLM.from_pretrained,
            "model_name": "xlm-roberta-base"       # model name for huggingface
        },
        "bloom": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": "bigscience/bloom-560m"       # model name for huggingface
        },
        "t0pp": {
            "model_constructor": AutoModelForSeq2SeqLM.from_pretrained,
            "model_name": 'bigscience/T0pp'  # model name for huggingface
        }
    }
}

task_config = {
    "DEFAULT_TASK": "SA",
    "SUPPORTED_TASKS": {
        "SA": {
            "label_map": {'5': 'yes', '4': 'yes', '3': 'yes', '2': 'no', '1': 'no',  '0': 'no'},
            "possible_answers": ['yes', 'no'],
            "prompt_class": SAPrompt
        },
        "NLI": {
            "label_map": {'0': 'yes', '1': 'maybe', '2': 'no'},
            "possible_answers": ['yes', 'no', 'maybe'],
            "prompt_class": NLIPrompt
        }  # ,
        # "NLI_v2": {
        #     "label_map": {'0': 'true', '1': 'neither', '2': 'false'},
        #     "possible_answers": ['true', 'false','neither'],
        #     "prompt_class": NLIPrompt_v2
        # }
    },
}
