# ATCS_group3
Research project for the cource ATCS at UvA. Our paper A Multilingual Perspective on Prompt Variability, looks into how certain grammatical properties of an instruction prompt can affect the performance of a model, and if this observation holds across languages. In our experiment we used the models Bloom, BloomZ, Flan-t5, mT0 and Llama and evaluated their performance on the tasks Sentiment Analysis (SA) and Natural Language Inference (NLI) when given prompts with different grammatical properties.

the grammatical properties compared in this experiment are:
- the use of active vs. passive prompts
- the use of auxillary vs. model verbs in the prompts
- the use of common words vs. rare synonyms (inspired by thesaurus)
- the use of different model verbs (non-binary comparison)

The novel part was that we than did this experiment across languages, so for German and French as well to check if the conclusions hold across languages.

A more in depth discription of the research can be found in our paper (report.pdf)


# How to run
1. Clone the repository
2. Create env for this project by running create_env.job. (if a package is still missing you can load it in with the help of install_pk.job)
3. To run the evaluation you can use run_pipeline.job, which can be given different args to specify which models, tasks, languages etc you want to evaluate.


'''
(srun) python ATCS_group3/src/main.py  --models bloom bloomz flan llama t0 --tasks SA NLI --languages en de fr --seeds 42 33 50 --sample_size 200 --version 96 --prompt_structure_name "prompt_structure_ABC_maybe" --answer_type_ABC True

'''



