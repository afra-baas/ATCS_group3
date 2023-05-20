# Config file
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from prompts.nli_prompt import NLIPrompt
from prompts.sa_prompt import SAPrompt

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
        "llama": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": "huggyllama/llama-7b"       # model name for huggingface
        },
        "bloom": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": "bigscience/bloom-560m"       # model name for huggingface
        },
        "bloomz": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": "bigscience/bloomz-560m"       # model name for huggingface
        },
        "alpaca": {
            "model_constructor": AutoModelForCausalLM.from_pretrained,
            "model_name": 'chainyo/alpaca-lora-7b'  # model name for huggingface
        },
        "flan": {
            "model_constructor": AutoModelForSeq2SeqLM.from_pretrained,
            "model_name": 'google/flan-t5-base'  # model name for huggingface
        }
    }
}

task_config = {
    "DEFAULT_TASK": "SA",
    "SUPPORTED_TASKS": {'en':
                        {"SA": {
                            # "label_map": {5: 'yes', 4: 'yes', 3: 'yes', 2: 'no', 1: 'no',  0: 'no'},
                            "label_map": {5: 'yes',  1: 'no'},
                            "possible_answers": ['yes', 'no'],
                            "prompt_class": SAPrompt
                        },
                            "NLI": {
                            "label_map": {0: 'yes', 1: 'maybe', 2: 'no'},
                            "possible_answers": ['yes', 'no', 'maybe'],
                            "prompt_class": NLIPrompt
                        }},
                        'de': {
                            "SA": {
                                "label_map": {5: 'ja',  1: 'nein'},
                                "possible_answers": ['ja', 'nein'],
                                "prompt_class": SAPrompt
                            },
                            "NLI": {
                                "label_map": {0: 'ja', 1: 'vielleicht', 2: 'nein'},
                                "possible_answers": ['ja', 'nein', 'vielleicht'],
                                "prompt_class": NLIPrompt
                            }},
                        'fr': {
                            "SA": {
                                "label_map": {5: 'oui',  1: 'non'},
                                "possible_answers": ['oui', 'non'],
                                "prompt_class": SAPrompt
                            },
                            "NLI": {
                                "label_map": {0: 'oui', 1: 'peut-être', 2: 'non'},
                                "possible_answers": ['oui', 'non', 'peut-être'],
                                "prompt_class": NLIPrompt
                            }}
                        }
}
