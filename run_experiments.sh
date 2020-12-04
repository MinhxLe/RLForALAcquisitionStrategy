python scripts/run_experiment.py --al_sampler="random"
python scripts/run_experiment.py --al_sampler="lc"
python scripts/run_experiment.py --al_sampler="ucb_bandit" --reward_metric_name='accuracy_metric'
# python scripts/run_experiment.py --al_sampler="ucb_bandit" --reward_metric_name='micro_f1_metric'

