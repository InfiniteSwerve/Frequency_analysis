from setuptools import setup, find_packages
setuptools.setup()
setup(
    name="frequency_analysis",
    version="0.1.0",
    packages=["frequency_analysis"],
    license="LICENSE",
    description="A helper library for understanding selection effects on transformers grokking modular arithmetic",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
