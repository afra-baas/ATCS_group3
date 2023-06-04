# ATCS_group3
research project for the cource ATCS at UvA.

# How to run
1. Clone the repository
2. Create env for this project by running create_env.job. (if a package is still missing you can load it in with the help of install_pk.job)
3. To run the evaluation you can use run_pipeline.job, which can be given different args to specify which models, tasks, languages etc you want to evaluate.


'''
(srun) python ATCS_group3/src/main.py  --models bloom bloomz flan llama t0 --tasks SA NLI --languages en de fr --seeds 42 33 50 --batch_size 16 --sample_size 200 --version 96 --module_name "prompt_structure_ABC_maybe" --answer_type_ABC True

'''



