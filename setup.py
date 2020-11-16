from setuptools import setup, find_packages

dependencies = [
    "torch",
    "jupyterlab",
    "pytest",
    "pytest-cov",
    "gensim",
    "pytorch-nlp",
    "matplotlib",
    "sklearn",
    "nltk",
    "tensorflow"
]

setup(
    name="src",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(include="bald"),
    install_requires=dependencies,
)
