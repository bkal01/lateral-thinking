#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting VM setup..."

# Update system packages
echo "📦 Updating system packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get upgrade -y
elif command -v yum &> /dev/null; then
    sudo yum update -y
elif command -v brew &> /dev/null; then
    brew update
fi

# Install Node.js and npm
echo "📦 Installing Node.js and npm..."
if command -v apt-get &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
elif command -v yum &> /dev/null; then
    curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
    sudo yum install -y nodejs npm
elif command -v brew &> /dev/null; then
    brew install node
fi

# Verify npm installation
npm --version

# Install Claude Code
echo "🤖 Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Install uv (Python package manager)
echo "🐍 Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Python dependencies using uv
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    uv pip install -e .
else
    echo "⚠️  No requirements.txt or pyproject.toml found, skipping Python dependencies"
fi

echo "✅ VM setup complete!"
echo "🔧 Installed tools:"
echo "  - Node.js: $(node --version)"
echo "  - npm: $(npm --version)"
echo "  - Claude Code: $(claude --version)"
echo "  - uv: $(uv --version)"