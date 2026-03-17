.PHONY: install train eval generate test ablation visualize clean

install:
	pip install -e ".[dev]"

prep:
	python scripts/train.py --config configs/base.yaml --phase prep

train:
	python scripts/train.py --config configs/base.yaml

train-baseline:
	python scripts/train.py --config configs/ablations/flat_baseline.yaml

eval:
	python scripts/eval.py --checkpoint checkpoints/best.pt

generate:
	python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

test:
	pytest tests/ -v

ablation:
	python scripts/ablation_run.py

visualize:
	python scripts/visualize.py --checkpoint checkpoints/best.pt

clean:
	rm -rf checkpoints/ wandb/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
