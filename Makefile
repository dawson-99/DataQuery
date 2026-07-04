.PHONY: test infra-test

test:
	.venv/bin/python -m pytest tests/ -v

infra-test:
	.venv/bin/python -m pytest tests/test_rule_review_infrastructure.py -v
