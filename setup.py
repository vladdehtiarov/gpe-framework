"""
GPE Framework Setup Script
==========================

Installation:
    pip install gpe-framework

Development installation:
    pip install -e .

Author: Vladyslav Dehtiarov
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="gpe-framework",
    version="1.0.0",
    author="Vladyslav Dehtiarov",
    author_email="vvdehtiarov@gmail.com",
    description="Greedy-Prune-Explain: Minimal local explanations for decision tree predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vdehtiarov/gpe-framework",
    project_urls={
        "Bug Tracker": "https://github.com/vdehtiarov/gpe-framework/issues",
        "Documentation": "https://github.com/vdehtiarov/gpe-framework#readme",
        "Source": "https://github.com/vdehtiarov/gpe-framework",
    },
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "experiments.*", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "baselines": [
            "lime>=0.2.0",
            "anchor-exp>=0.0.2",
        ],
        "all": [
            "lime>=0.2.0",
            "anchor-exp>=0.0.2",
            "pytest>=7.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "explainable-ai",
        "xai",
        "decision-trees",
        "interpretability",
        "machine-learning",
        "local-explanations",
        "model-explanation",
        "feature-importance",
    ],
)
