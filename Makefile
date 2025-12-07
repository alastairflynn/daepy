build:
	python -m build

.PHONY: upload
upload:
	twine upload --repository testpypi dist/*

.PHONY: publish
publish:
	twine upload dist/*

.PHONY: docs
docs:
	rm -rf build
	cd ./daepy/cheby && python setup.py build_ext --inplace
	cd ./docs && $(MAKE) html

.PHONY: test
test:
	cd ./test && python test_basic_usage.py
	cd ./test && python test_save_load.py
	cd ./test && python test_continuation.py

.PHONY: clean
clean:
	rm -rf dist
	rm -rf build
	rm -rf daepy/cheby/build
	rm daepy/cheby/cheby.cpython*.so

all: build
