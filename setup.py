from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SIREN",
    version="0.1.0",
    author="Yucong Zhang",
    author_email="yucong0428@outlook.com",
    description="Signal Representation Evaluation for Machines - A comprehensive evaluation framework for DCASE series datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yucongzh/SIREN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
        "thop>=0.1.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
)
