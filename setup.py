from setuptools import setup, find_packages

# TODO move dependencies here
dependencies = [
]

setup(
    name="src",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(include="src"),
    install_requires=dependencies,
)
