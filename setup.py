"""
Setup script for Shandu deep research system.
"""
from setuptools import setup, find_packages
import os

# Read long description from README.md
with open("shandu/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="shandu",
    version="0.1.1",
    description="Deep research system with LangChain and LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dušan Jolović",
    author_email="jolovic@pm.me",
    url="https://github.com/jolovicdev/shandu",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "shandu=shandu.cli:cli",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="research, ai, llm, langchain, langgraph, deepresearch, deepsearch, search",
    project_urls={
        "Source": "https://github.com/jolovicdev/shandu",
    },
    include_package_data=True,
)
