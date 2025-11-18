"""
Setup script for Quantitative Trading System package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quant-trading-system",
    version="3.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional-grade quantitative trading system with ML and portfolio optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-train=train:main",
            "quant-predict=predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "quantitative finance",
        "machine learning",
        "portfolio optimization",
        "algorithmic trading",
        "risk management",
        "factor models",
        "statistical arbitrage",
    ],
    project_urls={
        "Bug Reports": "https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning/issues",
        "Source": "https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning",
        "Documentation": "https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning/blob/main/README.md",
    },
)
