from setuptools import setup

setup(
    name="s4d",
    author="kavorite",
    license="MIT",
    description="structured state spaces for dm-haiku",
    find_packages={"": "src"},
    version="0.2.0",
    install_requires=["dm-haiku>=0.0.6", "einops~=0.4", "numpy~=1.22", "jax~=0.3"],
)
