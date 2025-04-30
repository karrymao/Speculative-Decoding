Use `python experiment.py` to run the experiments. 
The settings for your experiment can be set through `experiment_configs.json`. 
You can add more variables as long as the name of the variable is the same as function  arguments of `Experiment.__init__`. Also make sure that your new variable is saved to class attributes and is passed into `speculative_generate_multi()` in `_infer()`. \\
The result of the experiments will be **APPEND** to the result_file as specified in configs. 