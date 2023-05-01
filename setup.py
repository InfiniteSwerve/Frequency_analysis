from setuptools import setup, find_packages
setup(
    name="frequency-analysis",
    version="0.1.0",
    packages=["frequency_analysis"],
    license="LICENSE",
    description="A helper library for understanding selection effects on transformers grokking modular arithmetic",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "einops",
        "transformer_lens @ git+https://github.com/neelnanda-io/TransformerLens.git",
        "torch",
    ],
    extras_require={"dev": ["pytest", "mypy", "pytest-cov"]},
)
