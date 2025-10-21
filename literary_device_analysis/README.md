This analysis was done with a local clone of topicGPT with some small modifications, e.g., to allow specifying the number of GPUs available for local models. However, as the final analysis was done using GPT-4.1, those modifications should not be necessary to replicate the process. We also modified the `early_stop` parameter in the generation pipeline, setting it to 1000.

We ran `bl_generate.py` with the config file `config/config_humor_crowd.yml` to generate topics with input sentences from BL, crowd-first, and combo-humor. We did the same for refinement with `bl_refine.py`. We then ran `bl_assign.py` with each config file `config/config_humor_crowd_assign_*.yml` to do topic assignment for up to 1000 instances from each dataset.

We did not use the correction process with our topics.