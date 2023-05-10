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
├── data
│   └── test
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