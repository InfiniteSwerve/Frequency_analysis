from setuptools import setup

setup(
    name="frequency analysis",
    version="0.1.0",
    packages=["freq_analysis"],
    license="LICENSE",
    description="A helper library for understanding selection effects on transformers grokking modular arithmetic",
    long_description=open("README.md").read(),
    install_requires=[
        "einops",
        "transformer_lens @ git+https://github.com/neelnanda-io/TransformerLens.git",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "rich",
        "accelerate",
    ],
    extras_require={"dev": ["pytest", "mypy", "pytest-cov"]},
)
