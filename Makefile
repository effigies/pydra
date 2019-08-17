install:
	python setup.py install

dist: clean
	python setup.py sdist bdist_wheel

clean-pyc:
	find . -name '*.pyc' -type f -exec rm {} +
	find . -name '*.pyo' -type f -exec rm {} +
	find . -name '__pycache__' -type d -exec rm --recursive {} +

clean-build:
	rm --recursive --force build/
	rm --recursive --force dist/

clean: clean-pyc clean-build

format:
	black pydra tools setup.py

lint:
	# Run black twice to allow all issues to appear before failing
	black --check pydra tools setup.py || true
	flake8 pydra tools setup.py
	black --check pydra tools setup.py

test: clean-pyc
	py.test -vs -n auto --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra
