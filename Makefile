.PHONY: serve client eval eval-quick perf guardrails improve clean

serve:
	python serve/serve.py

client:
	python serve/client.py

eval:
	python eval_runner/run_eval.py

eval-quick:
	python eval_runner/run_eval.py --tasks hellaswag,mmlu,logical_reasoning --limit 50

perf:
	python perf/load_test.py

guardrails:
	python guardrails/validate.py

improve:
	bash improve/eval.sh hellaswag --limit 200

clean:
	rm -f eval_runner/.eval_cache.db
	rm -rf eval_runner/results/*.json
	rm -rf improve/data/ improve/predictions/
	rm -f perf/metrics.csv guardrails/report.json