# ATCS_group3
This research project, conducted as part of the ATCS course at the University of Amsterdam (UvA), explores the impact of various grammatical properties of instruction prompts on the performance of different language models. The aim is to investigate whether these observations remain consistent across different languages. The study evaluates the performance of five models, namely Bloom, BloomZ, Flan-t5, mT0, and Llama, on two tasks: Sentiment Analysis (SA) and Natural Language Inference (NLI).

### Research Focus
The primary focus of this research is to analyze the effects of specific grammatical properties on language model performance. The following grammatical properties were compared in the experiment:

- Active vs. Passive Prompts: A comparison on how using active or passive voice in prompts affects the models' performance.
- Auxiliary vs. Modal Verbs: A comparison is made between prompts that utilize auxiliary verbs and those that employ modal verbs.
- Common Words vs. Rare Synonyms: The impact of using common words versus using rare synonyms (from a thesaurus) in the prompts is compared.
- Different Modal Verbs (Non-Binary Comparison): This part of the study explores the effects of employing different modal verbs in the prompts.

### Multilingual Perspective
The novel aspect of this research project is the inclusion of multiple languages. In addition to English, the study extends its analysis to German and French to assess whether the findings hold true across different languages. By conducting the experiment in multiple languages, the research aims to determine the generalizability of the observed conclusions.

For a more detailed and comprehensive account of the research project, including the experimental design, methodology, and results, please refer to our paper titled "A Multilingual Perspective on Prompt Variability" (report.pdf). 

# How to run
### 1. Clone the Repository
Clone the repository to your local machine using the following command:
```
git clone https://github.com/afra-baas/ATCS_group3.git
```

### 2. Set Up the Environment
Create the environment required for this project by running the `create_env.job` script. This script will set up the necessary dependencies and packages. If any packages are still missing, you can use the `install_pk.job` script to install them.

### 3. Run the Evaluation
To run the evaluation, use the run_pipeline.job script. You can specify different arguments to customize the evaluation according to your preferences, such as the models, tasks, languages, seeds, sample size, version, prompt structure name, and answer type.

```
(srun) python ATCS_group3/src/main.py  --models bloom bloomz flan llama t0 --tasks SA NLI --languages en de fr --seeds 42 33 50 --sample_size 200 --version 96 --prompt_structure_name "prompt_structure_ABC_maybe" --answer_type_ABC True
```

### 4. Visualization of Results
The pickles generated during the evaluation will be saved. To visualize the results, you can use our demo.ipynb Colab notebook. You can access the notebook using the following link: [Demo Notebook](https://colab.research.google.com/drive/1fDeS0lVPl8urW68-Zs9VGZZQ5t6BrrLq#scrollTo=oD_Z30TwEOUH)
The notebook provides instructions and a convenient way to visualize and analyze the results obtained from the evaluation.
Download the "final_results" folder from this repository and upload it to your Google Drive. Follow the instructions provided in the demo notebook to load the results from the "final_results" folder. You can also upload your own result pickles to your Google Drive and visualize them using the same notebook.

### Individual Contributions
General contributions of each of our team members:

- Afra: codebase, unit tests, debugging, experiments/ablation studies, results notebook, research, poster, writing paper
- Amity: codebase, unit tests, debugging, research, poster, writing paper
- Emile: codebase, experiments, prompt design, debugging, demo notebook, writing paper
- Fabian: prompt design, notebook text, research, poster, writing paper
- Karsten: codebase, debugging, ablation studies, research, poster, writing paper
