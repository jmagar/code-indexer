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

# Repository information
REPO_URL="https://github.com/jmagar/code-indexer.git"
INSTALL_DIR="code-indexer"

# Function to check dependencies
check_dependencies() {
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose by trying to run it
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose (V2) is not available. Please install Docker Compose V2."
        exit 1
    fi
}

# Function to update the repository
update_repo() {
    if [ ! -d "$INSTALL_DIR" ]; then
        print_error "Code Indexer is not installed. Please run the install script without --update first."
        exit 1
    fi

    print_step "Updating Code Indexer..."
    cd "$INSTALL_DIR" || exit 1
    
    # Fetch latest changes
    git fetch origin main
    
    # Check if we're behind the remote
    if [ "$(git rev-list HEAD..origin/main --count)" != "0" ]; then
        print_step "New updates available, applying..."
        
        # Stash any local changes
        if ! git diff --quiet; then
            print_warning "Stashing local changes..."
            git stash
        fi
        
        # Pull latest changes
        git pull origin main
        
        # Update dependencies
        print_step "Updating dependencies..."
        # shellcheck source=/dev/null
        source .venv/bin/activate
        uv pip install -r requirements.txt
        
        print_step "Update complete! 🎉"
        echo -e "${GREEN}Code Indexer has been updated to the latest version${NC}"
    else
        echo -e "${GREEN}Code Indexer is already up to date!${NC}"
    fi
    exit 0
}

# Parse command line arguments
if [ "$1" == "--update" ]; then
    update_repo
fi

# Main installation
check_dependencies

if [ -d "$INSTALL_DIR" ]; then
    print_warning "Existing installation found at ./$INSTALL_DIR"
    print_warning "To update, run: $0 --update"
    exit 1
fi

print_step "Installing Code Indexer..."

# Clone repository
print_step "Cloning repository..."
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR" || exit 1

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

# Start Qdrant using Docker Compose
print_step "Starting Qdrant..."
if [ ! -f .env ]; then
    print_step "Creating .env file..."
    if [ ! -f .env.example ]; then
        echo "Error: .env.example file not found"
        exit 1
    fi
    cp .env.example .env
    echo "Please edit .env file with your API keys before continuing"
    exit 0
fi

docker compose up -d qdrant

print_step "Waiting for Qdrant to start..."
sleep 5

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
echo -e "${GREEN}You can now:"
echo -e "1. Edit .env file with your API keys (if you haven't already)"
echo -e "2. Restart your terminal or run: source $SHELL_RC"
echo -e "3. Run the indexer using the 'index' command:"
echo -e "   index search \"your query\""
echo -e "   index ingest github https://github.com/user/repo"
echo -e "   index ingest local /path/to/code"
echo -e "\nTo update in the future, run: ./scripts/install.sh --update${NC}" 