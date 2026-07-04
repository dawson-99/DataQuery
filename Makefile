.PHONY: test infra-test rewrite-test

test:
	.venv/bin/python -m pytest tests/ -v

infra-test:
	.venv/bin/python -m pytest tests/test_rule_review_infrastructure.py -v

rewrite-test:
	.venv/bin/python -m pytest tests/test_rule_review_query_rewriter.py -v
