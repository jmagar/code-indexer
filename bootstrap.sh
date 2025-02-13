#!/bin/bash
set -e

# Version
VERSION="1.0.0"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check current directory
if [ ! -w "$(pwd)" ]; then
    print_error "Current directory is not writable. Please run from a directory where you have write permissions."
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if directory exists
if [ -d "code-indexer" ]; then
    print_warning "Directory 'code-indexer' already exists. Please remove or rename it first."
    exit 1
fi

print_step "Code Indexer Bootstrap v${VERSION}"

# Clone repository
print_step "Cloning Code Indexer repository..."
if ! git clone https://github.com/jmagar/code-indexer.git; then
    print_error "Failed to clone repository. Please check your internet connection and try again."
    exit 1
fi

cd code-indexer || {
    print_error "Failed to enter code-indexer directory."
    exit 1
}

# Make scripts executable
chmod +x scripts/*.sh || {
    print_error "Failed to make scripts executable."
    exit 1
}

# Run installation script
print_step "Running installation script..."
./scripts/install.sh

print_step "Bootstrap complete! 🎉"
echo -e "${GREEN}The Code Indexer has been installed successfully."
echo -e "Follow the instructions above to complete the setup.${NC}" 