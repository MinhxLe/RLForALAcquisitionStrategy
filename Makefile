.PHONY: clean_results, clean_recent_experiment

clean_results:
	rm -rf results/*

clean_recent_experiment:
	ls -td results/* | head -n 1 | xargs rm -r
