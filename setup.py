from setuptools import setup, find_packages

setup(
    name="plexe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.24.0",
        "asyncio>=3.4.3",
    ],
    author="Plexe AI",
    author_email="support@plexe.ai",
    description="Create ML models from natural language descriptions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/plexe-ai/plexe",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ]
    },
)