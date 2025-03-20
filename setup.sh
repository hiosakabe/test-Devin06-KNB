#!/bin/bash

# Setup script for racing data analysis environment

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "pyenv is not installed. Please install pyenv first."
    echo "Visit: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install Poetry first."
    echo "Visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install Python version if not already installed
if ! pyenv versions | grep -q 3.12.0; then
    echo "Installing Python 3.12.0..."
    pyenv install 3.12.0
fi

# Set local Python version
echo "Setting local Python version to 3.12.0..."
pyenv local 3.12.0

# Install dependencies with Poetry
echo "Installing dependencies with Poetry..."
poetry install

echo "Setup complete! You can now activate the virtual environment with 'poetry shell'"
