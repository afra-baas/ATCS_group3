# Config file
data = {
    "NLI": {
        "dataset": "NLI",
        "language": "French",
        "dataset_type": "train",
        "batch_size": 32,
    },
    "MARC": {
        "dataset": "MARC",
        'path': './data/marc_data/',
        "language": "English",
        "dataset_type": "train",
        "batch_size": 32,
    }
}

model = {
    "DEFAULT_MODEL": "xlm-roberta-base",
    "SUPPORTED_MODELS": ["xlm-roberta-base", "bigscience/bloom-560m"],
}

task = {
    "DEFAULT_TASK": "SA",
    "SUPPORTED_TASKS": ["SA", "NLI"],
}
