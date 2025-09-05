# ChromaDB Server Deployment Guide

> **Transform your ChromaDB from local to scalable server architecture**

This guide helps you deploy ChromaDB as a centralized server with HTTP API access, enabling multiple clients to share the same vector database for your AI Meetup Recommendation system.

## 🏗️ Architecture Overview

```
┌─────────────────┐    HTTP API    ┌──────────────────┐    Direct Access    ┌──────────────────┐
│   AI Clients    │ ──────────────► │  ChromaDB Server │ ──────────────────► │   ChromaDB Core  │
│                 │                 │   (FastAPI)      │                     │  (Vector Store)   │
│ • Bot Instance  │                 │                  │                     │                  │
│ • Web App       │                 │ • REST Endpoints │                     │ • Events Data    │
│ • Mobile App    │                 │ • Authentication │                     │ • User Prefs     │
│ • Dashboard     │                 │ • Rate Limiting  │                     │ • Embeddings     │
└─────────────────┘                 └──────────────────┘                     └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 4GB+ RAM, 2+ CPU cores
- Domain name or static IP (for production)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd ai-agents-chromadb-server
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env file with your configuration
```

### 3. Deploy with Docker
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Verify Deployment
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", "version": "1.0.0"}
```

## 📁 Project Structure

```
chromadb-server/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 Dockerfile                   # Container configuration
├── 📄 docker-compose.yml           # Development setup
├── 📄 docker-compose.prod.yml      # Production setup
├── 📄 .env.example                 # Environment template
├── 📄 nginx.conf                   # Reverse proxy config
├── 
├── 📂 server/
│   ├── 📄 __init__.py
│   ├── 📄 main.py                  # FastAPI server entry point
│   ├── 📄 chromadb_server.py       # Core ChromaDB API endpoints
│   ├── 📄 models.py                # Pydantic models
│   ├── 📄 auth.py                  # Authentication logic
│   ├── 📄 config.py                # Configuration management
│   └── 📄 utils.py                 # Utility functions
├── 
├── 📂 client/
│   ├── 📄 __init__.py
│   ├── 📄 chromadb_client.py       # HTTP API client wrapper
│   ├── 📄 exceptions.py            # Custom exceptions
│   └── 📄 models.py                # Client-side models
├── 
├── 📂 migration/
│   ├── 📄 migrate_data.py           # Data migration script
│   ├── 📄 export_local.py          # Export from local ChromaDB
│   └── 📄 import_remote.py         # Import to remote ChromaDB
├── 
├── 📂 tests/
│   ├── 📄 test_server.py           # Server endpoint tests
│   ├── 📄 test_client.py           # Client wrapper tests
│   └── 📄 test_integration.py      # End-to-end tests
├── 
└── 📂 scripts/
    ├── 📄 deploy.sh                # Deployment script
    ├── 📄 backup.sh                # Data backup script
    └── 📄 monitor.sh               # Health monitoring
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Server Configuration
CHROMADB_SERVER_HOST=0.0.0.0
CHROMADB_SERVER_PORT=8000
CHROMADB_DATA_PATH=/app/data

# Security
API_KEY=your-super-secure-api-key-here
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
RATE_LIMIT_PER_MINUTE=60

# ChromaDB Settings
CHROMADB_PERSIST_DIRECTORY=/app/chromadb_data
EMBEDDING_MODEL=all-mpnet-base-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/chromadb-server.log

# Production Settings (optional)
SSL_CERT_PATH=/app/certs/cert.pem
SSL_KEY_PATH=/app/certs/key.pem
```

## 📡 API Reference

### Core Collections API

#### Create Collection
```http
POST /api/v1/collections
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "name": "meetup_events",
    "metadata": {"description": "Meetup events for AI recommendations"}
}
```

#### List Collections
```http
GET /api/v1/collections
Authorization: Bearer your-api-key
```

#### Get Collection Info
```http
GET /api/v1/collections/meetup_events
Authorization: Bearer your-api-key
```

### Document Operations

#### Add Documents
```http
POST /api/v1/collections/meetup_events/add
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "documents": ["Event description text..."],
    "metadatas": [{"event_id": "123", "name": "Tech Meetup"}],
    "ids": ["event_123"]
}
```

#### Query Documents
```http
POST /api/v1/collections/meetup_events/query
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "query_texts": ["looking for tech events"],
    "n_results": 5,
    "where": {"activity": "Technology"}
}
```

#### Get Documents
```http
GET /api/v1/collections/meetup_events/get?ids=event_123,event_124
Authorization: Bearer your-api-key
```

#### Update Documents
```http
PUT /api/v1/collections/meetup_events/update
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "ids": ["event_123"],
    "documents": ["Updated event description"],
    "metadatas": [{"event_id": "123", "name": "Updated Tech Meetup"}]
}
```

### Event-Specific Endpoints

#### Bulk Sync Events
```http
POST /api/v1/events/sync
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "events": [...],
    "clear_existing": true
}
```

#### Search Events
```http
POST /api/v1/events/search
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "query": "football events in Mumbai",
    "n_results": 10,
    "filters": {"city_name": "Mumbai"}
}
```

#### Get Event by ID
```http
GET /api/v1/events/123
Authorization: Bearer your-api-key
```

### User Preferences Endpoints

#### Add User Preferences
```http
POST /api/v1/user-preferences/bulk
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "preferences": [...],
    "clear_existing": false
}
```

#### Get User Preferences
```http
GET /api/v1/user-preferences/user_456
Authorization: Bearer your-api-key
```

### Health & Monitoring

#### Health Check
```http
GET /health
```

#### Metrics
```http
GET /metrics
Authorization: Bearer your-api-key
```

## 💻 Client Integration

### Python Client Usage
```python
from client.chromadb_client import ChromaDBClient

# Initialize client
client = ChromaDBClient(
    server_url="http://localhost:8000",
    api_key="your-api-key"
)

# Search events
results = client.search_events(
    query="tech meetup",
    n_results=5,
    filters={"city_name": "Mumbai"}
)

# Add events
client.add_events(events_data, clear_existing=True)

# Get user preferences
prefs = client.get_user_preferences("user_123")
```

### Update Existing Code
Replace your existing `ChromaDBManager` initialization:

```python
# Before (local ChromaDB)
chroma_manager = ChromaDBManager()

# After (remote ChromaDB server)
chroma_manager = ChromaDBManager(
    server_url="http://your-chromadb-server:8000",
    api_key="your-api-key"
)
```

## 🚢 Deployment Options

### 1. Development (Local)
```bash
# Clone and setup
git clone <repo-url>
cd chromadb-server
cp .env.example .env

# Run with Docker Compose
docker-compose up -d

# Access at: http://localhost:8000
```

### 2. Production (Cloud)

#### Docker Deployment
```bash
# Build and deploy
docker build -t chromadb-server:latest .
docker run -d \
  --name chromadb-server \
  -p 8000:8000 \
  -v /data/chromadb:/app/data \
  -e API_KEY=your-secure-key \
  chromadb-server:latest
```

#### Docker Compose (Recommended)
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  chromadb-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - chromadb-server
    restart: unless-stopped
```

#### Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chromadb-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: chromadb-server
  template:
    metadata:
      labels:
        app: chromadb-server
    spec:
      containers:
      - name: chromadb-server
        image: chromadb-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: chromadb-secret
              key: api-key
```

## 🔐 Security Best Practices

### 1. API Key Management
```bash
# Generate secure API key
openssl rand -hex 32

# Store in environment variables
export CHROMADB_API_KEY="your-generated-key"
```

### 2. HTTPS Configuration
```bash
# Generate SSL certificates (Let's Encrypt)
certbot --nginx -d your-domain.com

# Or use self-signed for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### 3. Network Security
```bash
# Firewall rules (Ubuntu/Debian)
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable

# Only allow specific IPs (optional)
ufw allow from YOUR.CLIENT.IP.ADDRESS to any port 8000
```

### 4. Rate Limiting
```python
# Built-in rate limiting (60 requests/minute per IP)
# Configure in .env:
RATE_LIMIT_PER_MINUTE=60
```

## 📊 Monitoring & Logging

### Application Logs
```bash
# View logs
docker-compose logs -f chromadb-server

# Log files location
tail -f ./logs/chromadb-server.log
```

### Health Monitoring
```bash
# Health check endpoint
curl http://localhost:8000/health

# Metrics endpoint
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/metrics
```

### Performance Monitoring
```python
# Built-in metrics:
# - Request count and latency
# - Collection sizes
# - Query performance
# - Memory usage
# - Error rates
```

## 🔄 Data Migration

### Export from Local ChromaDB
```python
python migration/export_local.py \
  --source ./local_chromadb_data \
  --output ./exported_data.json
```

### Import to Remote ChromaDB
```python
python migration/import_remote.py \
  --server-url http://your-server:8000 \
  --api-key your-api-key \
  --data ./exported_data.json
```

### Full Migration Script
```bash
# Automated migration
python migration/migrate_data.py \
  --source-local ./local_data \
  --target-server http://your-server:8000 \
  --api-key your-api-key \
  --verify
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/test_server.py -v

# Integration tests
pytest tests/test_integration.py -v

# All tests with coverage
pytest tests/ --cov=server --cov-report=html
```

### Manual API Testing
```bash
# Test server health
curl http://localhost:8000/health

# Test authentication
curl -H "Authorization: Bearer wrong-key" \
     http://localhost:8000/api/v1/collections
# Should return 401 Unauthorized

# Test valid request
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/api/v1/collections
# Should return collections list
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Check if server is running
docker ps | grep chromadb-server

# Check logs for errors
docker logs chromadb-server

# Verify port binding
netstat -tlnp | grep :8000
```

#### 2. Authentication Errors
```bash
# Verify API key in environment
echo $API_KEY

# Check request headers
curl -v -H "Authorization: Bearer your-api-key" http://localhost:8000/health
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats chromadb-server

# Increase memory limits
docker run -m 4g chromadb-server:latest
```

#### 4. Data Persistence Issues
```bash
# Check volume mounts
docker inspect chromadb-server | grep -A 10 "Mounts"

# Verify data directory permissions
ls -la ./data/
```

### Performance Optimization

#### 1. Increase Worker Processes
```yaml
# docker-compose.yml
environment:
  - WORKERS=4  # Adjust based on CPU cores
```

#### 2. Optimize ChromaDB Settings
```python
# config.py
CHROMADB_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": False,
    "persist_directory": "/app/chromadb_data"
}
```

#### 3. Add Redis Caching
```yaml
# Add to docker-compose.yml
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Deploy multiple server instances
- Use load balancer (Nginx, HAProxy)
- Implement session affinity if needed

### Vertical Scaling
- Increase CPU and RAM
- Use SSD storage for better I/O
- Optimize embedding model size

### Database Sharding
- Separate collections by domain
- Route requests based on collection type
- Implement data partitioning strategies

## 🔄 Backup & Recovery

### Automated Backup
```bash
# Daily backup script (backup.sh)
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_${DATE}.tar.gz" ./data/
aws s3 cp "backup_${DATE}.tar.gz" s3://your-backup-bucket/
```

### Recovery Process
```bash
# Download backup
aws s3 cp s3://your-backup-bucket/backup_20241201_120000.tar.gz ./

# Restore data
tar -xzf backup_20241201_120000.tar.gz
docker-compose down
docker-compose up -d
```

## 📋 Maintenance Checklist

### Daily
- [ ] Check server health status
- [ ] Monitor error logs
- [ ] Verify backup completion

### Weekly  
- [ ] Review performance metrics
- [ ] Update security patches
- [ ] Clean up old log files

### Monthly
- [ ] Test backup recovery
- [ ] Review access logs
- [ ] Update dependencies

## 🆘 Support & Contributing

### Getting Help
- 📧 Email: support@yourdomain.com
- 💬 Discord: [Your Discord Server]
- 📚 Wiki: [Documentation Wiki]
- 🐛 Issues: [GitHub Issues]

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏁 Quick Reference

### Essential Commands
```bash
# Start server
docker-compose up -d

# View logs  
docker-compose logs -f

# Restart server
docker-compose restart

# Stop server
docker-compose down

# Health check
curl http://localhost:8000/health

# API test
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/collections
```

### Configuration Files
- `.env` - Environment variables
- `docker-compose.yml` - Development setup
- `docker-compose.prod.yml` - Production setup
- `nginx.conf` - Reverse proxy configuration

### Important URLs
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

---

**Ready to deploy ChromaDB as a scalable server? Follow this guide step by step and transform your local vector database into a production-ready service! 🚀**