.PHONY: lint precommit
lint:
    @flake8 --exclude=.tox

.PHONY: run
run: 
	@python3 -m jupyter notebook

precommit:
	# running the server dirties commits
	@echo 'Checking out all unsaved files'
	@git checkout -- Homework/*
