# ATCS_group3
research project for the cource ATCS at UvA

# How to run
1. Clone the repository
2. Install the requirements using poetry:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    poetry install
    ```
3. Run the repositories using `poetry run my-script`
   
# File structure
```
.
├── README.md
├── models
├── notebooks
├── output
└── src
    ├── __init__.py
    ├── config.py
    ├── main.py
    ├── evaulate.py
    ├── data                   # data processing - loading MARC from path and NLI from huggingface
    │   ├── __init__.py
    │   ├── hf_dataloder.py
    │   ├── hf_dataset.py
    │   ├── MARC
    │   │   ├── __init__.py
    │   │   ├── dataloader.py
    │   │   └── dataset.py
    │   └── NLI
    │       ├── __init__.py 
    │       ├── dataloader.py
    │       └── dataset.py
    ├── models                # Loading model from huggingface
    │   ├── __init__.py
    │   └──  model.py
    ├── prompts               # Generating prompts using the given template for each sentence
    │   ├── __init__.py
    │   └──  prompt.py        # A prompt class that generates prompts from a template
    ├── visualization
    │   ├── __init__.py
    │   └── visualize.py
    └── utils
        ├── __init__.py
        ├── config.py
        ├── logger.py
        └── utils.py
```

# Structure of the data
The data is pulled from Huggingface datasets. The data is split into train, validation and test sets. The data is split into 2 different datasets:
1. MARC
2. NLI
Both datasets are passed to a dataloader which
1. Loads the data
2. Collates the data into batches of size `batch_size` (default: 32)
3. Each batch is a list of tuples (sentences, label) where `sentences` is a list of sentences and `label` is a list of labels for each sentence in `sentences`

# Defining a new prompt
Each dataloader has a list of supported tasks. Tasks
are associated with particular dataloaders because
the task meaning is provided by the amount of sentences
the dataloader sends to it and the labels it receives.
A task contains two elements:
1. A class of the prompt to be used. The prompt is used
by the dataloader during preprocessing. Per
batch item, the prompt takes the item's list of sentences
and returns a string.
2. A dictionary that connects the labels to meaningful names.

To add a new task and prompt, add a new entry to the `task_config` list
in the config file. The entry should be a dictionary with the following
keys:
1. `label_map`: A dictionary that maps the labels to meaningful names.
2. `prompt`: A class that inherits from `Prompt` and implements the `generate_prompt` method.

To define a new prompt, create a new class that inherits from `Prompt` and
contains:
1. `prompt_instructions` class variable: A string that describes the prompt instructions.
2. `prompt_query` class variable: A string that describes the prompt query.
3. `__call__` method: Takes a list of sentences and returns a string.