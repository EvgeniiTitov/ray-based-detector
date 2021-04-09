.PHONY: help
help:
	@cat $(MAKEFILE_LIST) | grep -e "^[a-zA-Z0-9_\-]*: *.*## *" | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:
	pip install --upgrade pip
	pip install -r requirements.txt

pre-commit-install:
	pre-commit install
