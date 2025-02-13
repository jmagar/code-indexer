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

# Generate a secure random API key
generate_api_key() {
    openssl rand -base64 32 | tr -d '/+=' | cut -c1-32
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Set Qdrant host and port
QDRANT_HOST="localhost"
QDRANT_PORT="6550"
QDRANT_URL="http://${QDRANT_HOST}:${QDRANT_PORT}"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    print_step "Generating new .env file..."
    
    # Generate Qdrant API key
    QDRANT_KEY=$(generate_api_key)
    
    # Create .env with generated keys
    cat > .env <<EOL
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here  # Required for embeddings
QDRANT_API_KEY=${QDRANT_KEY}             # Auto-generated

# Optional: GitHub Integration
GITHUB_TOKEN=your_github_token_here      # Optional: For GitHub repository access
GITHUB_WEBHOOK_SECRET=your_secret_here   # Optional: For GitHub webhooks

# Qdrant Configuration
QDRANT_HOST=${QDRANT_HOST}
QDRANT_PORT=${QDRANT_PORT}
QDRANT_URL=${QDRANT_URL}

# Optional: Processing Configuration
CHUNK_SIZE=750
CHUNK_OVERLAP=100
BATCH_SIZE=100
MIN_SCORE=0.7

# Optional: Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
EOL

    print_step "Created new .env file with secure Qdrant API key"
    print_warning "You MUST set your OpenAI API key in .env before using the indexer"
    echo -e "   Get your key at: ${GREEN}https://platform.openai.com/api-keys${NC}"
    
    print_step "Optional: Set GitHub token for repository access"
    echo -e "   Get your token at: ${GREEN}https://github.com/settings/tokens${NC}"
else
    print_step "Using existing .env file"
fi

# Source the .env file to get the API key
# shellcheck source=/dev/null
source .env

# Start Qdrant
print_step "Starting Qdrant..."
docker compose up -d qdrant

# Wait for Qdrant to be ready
print_step "Waiting for Qdrant to start..."
max_attempts=30
attempt=1
while ! curl -s -f -H "api-key: ${QDRANT_API_KEY}" "${QDRANT_URL}/collections" > /dev/null; do
    if [ $attempt -eq $max_attempts ]; then
        print_error "Qdrant failed to start after ${max_attempts} attempts"
        exit 1
    fi
    echo -n "."
    sleep 1
    ((attempt++))
done
echo ""

# Get container info
CONTAINER_ID=$(docker ps -qf "name=index-db")
if [ -z "$CONTAINER_ID" ]; then
    print_error "Could not find Qdrant container"
    exit 1
fi

# Get container IP
CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER_ID")

# Get Qdrant info and stats
print_step "Qdrant is running! 🚀"
echo -e "${GREEN}Access Details:${NC}"
echo -e "Local URL: ${QDRANT_URL}"
echo -e "Container IP: ${CONTAINER_IP}"
echo -e "Container Name: index-db"
echo -e "API Key: ${QDRANT_API_KEY}"

print_step "Checking Qdrant collections..."
COLLECTIONS=$(curl -s -H "api-key: ${QDRANT_API_KEY}" "${QDRANT_URL}/collections")
if [ "$(echo "$COLLECTIONS" | jq '.result | length')" -eq 0 ]; then
    echo "No collections found - ready for initialization!"
else
    echo "Existing collections:"
    echo "$COLLECTIONS" | jq -r '.result[] | "- \(.name) (\(.vectors_count) vectors)"'
fi

# Check disk usage
STORAGE_SIZE=$(docker exec "$CONTAINER_ID" du -sh /qdrant/storage 2>/dev/null | cut -f1)
echo -e "\nStorage usage: ${STORAGE_SIZE}"

print_step "Configuration Status:"
echo -e "\n1. Required Configuration:"
if [ ! -f .env ] || grep -q "your_openai_api_key_here" .env; then
    echo -e "${RED}✗ OpenAI API Key:${NC} Not configured"
    echo -e "  → Edit .env and set your OPENAI_API_KEY"
    echo -e "  → Get your key at: ${GREEN}https://platform.openai.com/api-keys${NC}"
else
    echo -e "${GREEN}✓ OpenAI API Key:${NC} Configured"
fi

echo -e "\n2. Optional Configuration:"
if grep -q "your_github_token_here" .env; then
    echo -e "${YELLOW}○ GitHub Integration:${NC} Not configured"
    echo -e "  → For GitHub repository access:"
    echo -e "    1. Get a token at: ${GREEN}https://github.com/settings/tokens${NC}"
    echo -e "    2. Edit .env and set GITHUB_TOKEN"
    echo -e "    3. (Optional) Set GITHUB_WEBHOOK_SECRET for webhooks"
else
    echo -e "${GREEN}✓ GitHub Integration:${NC} Configured"
fi

echo -e "\n3. Quick Start Examples:"
echo -e "${BLUE}Local Code Indexing:${NC}"
echo -e "   ${GREEN}index ingest local ./my-project${NC}"
echo -e "   ${GREEN}index ingest local /path/to/code --exclude 'node_modules,dist'${NC}"

echo -e "\n${BLUE}GitHub Repository Indexing:${NC}"
echo -e "   ${GREEN}index ingest github https://github.com/username/repo${NC}"
echo -e "   ${GREEN}index ingest github org/repo --branch main${NC}"

echo -e "\n${BLUE}Searching Code:${NC}"
echo -e "   ${GREEN}index search \"how to connect to database\"${NC}"
echo -e "   ${GREEN}index search \"error handling\" --paths src/utils/errors${NC}"
echo -e "   ${GREEN}index search \"api endpoints\" --min-score 0.8${NC}"

echo -e "\n${BLUE}Monitoring:${NC}"
echo -e "   ${GREEN}index status${NC}         # Show indexing status"
echo -e "   ${GREEN}index collections${NC}    # List collections"
echo -e "   ${GREEN}index stats${NC}          # Show usage statistics" 