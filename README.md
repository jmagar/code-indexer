# Code Indexer and Semantic Search

A powerful tool for indexing and semantically searching codebases using embeddings and vector search. Supports both local directories and GitHub repositories with real-time updates.

## Features

- 🔍 **Semantic Code Search**: Find code by meaning, not just text matching
- 🔄 **Multiple Sources**: 
  - Local code directories
  - GitHub repositories with webhook support
- 🤖 **Multiple Embedding Providers**:
  - OpenAI (`text-embedding-ada-002`)
  - LM Studio (`bge-large-en-v1.5`)
- 📦 **Efficient Storage**:
  - Automated Qdrant deployment via Docker Compose
  - Secure API key generation
  - Persistent vector storage
  - Incremental updates
  - File change detection
- 🔧 **Advanced Features**:
  - Path filtering
  - Configurable similarity thresholds
  - Repository metadata tracking
  - Line number preservation
  - Real-time GitHub webhook integration

## Prerequisites

- Git
- Docker with Docker Compose V2
- Python 3.8+
- OpenAI API key (required for default embeddings)
- (Optional) GitHub token for repository access

## Quick Start

### One-Line Installation
```bash
curl -sSL https://raw.githubusercontent.com/jmagar/code-indexer/main/bootstrap.sh | bash
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/jmagar/code-indexer.git
cd code-indexer
```

2. Run the installation script:
```bash
./scripts/install.sh
```

The script will:
- Set up Python environment and dependencies
- Create necessary scripts and commands

3. Deploy Qdrant:
```bash
./scripts/deploy.sh
```

The deploy script will:
- Generate a secure random API key for Qdrant
- Create a `.env` file with necessary configuration
- Start Qdrant in Docker with persistent storage
- Create and verify a test collection
- Guide you through setting up required API keys

### Qdrant Configuration

The project uses [Qdrant](https://qdrant.tech/) as its vector database, deployed automatically via Docker Compose. Our setup provides:

- **Automated Deployment**: Single command deployment via `deploy.sh`
- **Security**: Auto-generated API keys and secure configuration
- **Persistence**: Docker volumes for reliable data storage
- **Isolation**: Runs on ports 6550/6551 to avoid conflicts
- **Health Checks**: Automatic verification of deployment
- **Monitoring**: Built-in collection statistics and status

The `.env` file contains all configuration options:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key    # Required for embeddings
QDRANT_API_KEY=auto_generated         # Generated during setup

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6550                     # REST API
QDRANT_URL=http://localhost:6550     # Base URL for API calls

# Optional: GitHub Integration
GITHUB_TOKEN=your_github_token          # For repository access
GITHUB_WEBHOOK_SECRET=your_secret       # For webhook security

# Processing Settings
CHUNK_SIZE=750                         # Code chunk size
CHUNK_OVERLAP=100                      # Overlap between chunks
BATCH_SIZE=100                         # Batch size for processing
MIN_SCORE=0.7                         # Minimum similarity score

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json                        # or 'text'
```

## Usage

### Local Code Indexing
```bash
# Index a project
index ingest local ./my-project

# Exclude specific directories
index ingest local /path/to/code --exclude 'node_modules,dist'
```

### GitHub Repository Indexing
```bash
# Index a repository
index ingest github https://github.com/username/repo

# Index specific branch
index ingest github org/repo --branch main
```

### Searching Code
```bash
# Basic search
index search "function to calculate fibonacci"

# Search with path filter
index search "error handling" --paths src/utils/errors

# Search with minimum similarity score
index search "api endpoints" --min-score 0.8
```

### Monitoring
```bash
# Show indexing status
index status

# List collections
index collections

# Show usage statistics
index stats
```

## Updating

To update to the latest version:

```bash
./scripts/install.sh --update
```

This will:
- Pull the latest changes
- Update dependencies
- Preserve your configuration

## Architecture

### Components

1. **CodeProcessor**: Main class handling code processing and indexing
   - File chunking
   - Embedding generation
   - Vector storage

2. **Plugins**:
   - `GitHubPlugin`: GitHub repository processing
   - `LocalCodePlugin`: Local directory processing
   - `QdrantSearchPlugin`: Vector similarity search

3. **Embedding Providers**:
   - `OpenAIEmbedding`: OpenAI's text-embedding-ada-002
   - `LMStudioEmbedding`: Local embedding using bge-large-en

4. **Infrastructure**:
   - `Docker Compose`: Automated Qdrant deployment
   - `deploy.sh`: Container and environment setup
   - `install.sh`: Python environment and dependency management

### Data Flow

1. Source code → Text chunks
2. Chunks → Embeddings
3. Embeddings → Vector database (Qdrant)
4. Search query → Query embedding
5. Query embedding → Similar code snippets

## Performance

- Processing speed: ~15 vectors/second
- Average indexing time: ~1.5-1.9s per 23 vectors
- Efficient batch processing (100 points per batch)
- Incremental updates for changed files only

## Security

- API keys stored in environment variables
- Secure random key generation for Qdrant
- Webhook signature verification
- Temporary file cleanup
- Safe file handling

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Check if Docker is running: `docker ps`
   - Verify Qdrant container status: `docker logs index-db`
   - Check container ports: `docker port index-db`
   - Confirm port availability: `netstat -an | grep 6550`
   - Restart container: `docker compose restart qdrant`
   - Check logs: `index logs`

2. **OpenAI API Issues**
   - Verify API key in `.env`
   - Check OpenAI service status
   - Monitor rate limits

3. **GitHub Integration**
   - Verify token permissions
   - Check repository access
   - Confirm webhook configuration

### Getting Help

- Check the logs: `index logs`
- Run with debug logging: `LOG_LEVEL=DEBUG index ...`
- Open an issue on GitHub

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenAI for embeddings API
- Qdrant for vector database
- GitHub for repository integration 