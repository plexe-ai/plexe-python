.PHONY: install clean build publish

install:
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Update version like: make version VERSION=0.1.1
version:
	sed -i 's/version="[0-9.]*"/version="$(VERSION)"/' setup.py