build: setup.py
	python setup.py sdist bdist_wheel

.PHONY: upload
upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish:
	twine upload dist/*

.PHONY: docs
docs:
	cd ./docs && $(MAKE) html

.PHONY: test
test:
	cd ./test && python test_basic_usage.py
	cd ./test && python test_continuation.py

.PHONY: clean
clean:
	rm dist/*

all:
	build
