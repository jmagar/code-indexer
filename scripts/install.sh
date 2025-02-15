#!/bin/bash
set -e

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

# Function to clean cache directories
clean_caches() {
    print_step "Cleaning cache directories..."
    rm -rf .mypy_cache/
    rm -rf .pytest_cache/
    rm -rf .ruff_cache/
    rm -rf .uv/
    rm -rf __pycache__/
    find . -type d -name "__pycache__" -exec rm -rf {} +
}

# Function to update dependencies
update_deps() {
    print_step "Updating dependencies..."
    # shellcheck source=/dev/null
    source .venv/bin/activate
    uv pip install -r requirements.txt
    clean_caches
}

# Parse command line arguments
if [ "$1" == "--update" ]; then
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found. Are you in the code-indexer directory?"
        exit 1
    fi
    update_deps
    print_step "Update complete! 🎉"
    echo -e "${GREEN}Dependencies have been updated to the latest version${NC}"
    exit 0
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Are you in the code-indexer directory?"
    exit 1
fi

print_step "Setting up Python environment..."

# Clean any existing cache directories
clean_caches

# Install uv if not already installed
print_step "Installing/updating uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for this session if not already there
export PATH="$HOME/.cargo/bin:$PATH"

# Create Python virtual environment using uv
print_step "Creating Python virtual environment..."
uv venv .venv

# Activate virtual environment
print_step "Activating virtual environment..."
# shellcheck source=/dev/null
source .venv/bin/activate

# Install dependencies using uv
print_step "Installing dependencies..."
uv pip install -r requirements.txt

# Create shell script for the index command
print_step "Creating index command..."
cat > scripts/index <<EOL
#!/bin/bash
source "\$(dirname "\$(dirname "\$(realpath "\$0")")")/.venv/bin/activate"
python "\$(dirname "\$(dirname "\$(realpath "\$0")")")/processor.py" "\$@"
EOL

chmod +x scripts/index

# Add to PATH if not already there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPT_PATH="$HOME/.local/bin"

mkdir -p "$SCRIPT_PATH"
ln -sf "$SCRIPT_DIR/index" "$SCRIPT_PATH/index"

# Detect shell and add PATH if needed
SHELL_RC=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
    if ! grep -q "PATH=\"\$HOME/.local/bin:\$PATH\"" "$SHELL_RC"; then
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
        print_step "Added ~/.local/bin to PATH in $SHELL_RC"
    fi
fi

print_step "Installation complete! 🎉"
echo -e "${GREEN}Next steps:"
echo -e "1. Run the deployment script to start Qdrant:"
echo -e "   ${BLUE}./scripts/deploy.sh${NC}"
echo -e "2. Edit .env file with your API keys"
echo -e "3. Restart your terminal or run: source $SHELL_RC"
echo -e "\nTo update dependencies in the future, run: ./scripts/install.sh --update${NC}" 