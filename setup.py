from setuptools import find_namespace_packages, setup

setup(
    name="s4d",
    author="kavorite",
    license="MIT",
    description="structured state spaces for dm-haiku",
    packages=find_namespace_packages(),
    version="0.3.0",
    install_requires=["dm-haiku", "einops", "numpy", "jax"],
)
