#!/bin/bash

# Check if 'uv' command is installed
if ! command -v uv &> /dev/null; then
    echo "'uv' command could not be found. Installing now..."
    
    if command -v curl &> /dev/null; then
        echo "'curl' is installed. Using it to download and install 'uv'."
        sh <(curl -LsSf https://astral.sh/uv/install.sh)
    elif command -v wget &> /dev/null; then
        echo "'wget' is installed. Using it to download and install 'uv'."
        sh <(wget -qO- https://astral.sh/uv/install.sh)
    else
        echo "Neither 'curl' nor 'wget' is installed. Please install one of them first."
        exit
    fi
    
    if [ $? -eq 0 ]; then
        echo "'uv' has been successfully installed."
    else
        echo "Failed to install 'uv'. Please check your package manager and try again."
        exit 1
    fi
else
    echo "'uv' command is already installed."
fi

