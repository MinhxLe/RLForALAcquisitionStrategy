.PHONY: clean_recent_experiment, clean_debug


clean_recent_experiment:
	ls -td results/* | head -n 1 | xargs rm -r

clean_debug:
	rm -rf results/DEBUG/*
