1#!/bin/bash

# Build and push Docker image to Docker Hub
# Usage: ./build-push.sh <dockerhub-username> <version>
# Example: ./build-push.sh myusername v1.0.0

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -ne 2 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 <dockerhub-username> <version>"
    echo "Example: $0 myusername v1.0.0"
    exit 1
fi

DOCKER_USERNAME=$1
VERSION=$2
IMAGE_NAME="ai-agent"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo -e "${GREEN}=== AI Agent Docker Build & Push ===${NC}"
echo -e "Docker Hub User: ${YELLOW}${DOCKER_USERNAME}${NC}"
echo -e "Version: ${YELLOW}${VERSION}${NC}"
echo -e "Image: ${YELLOW}${FULL_IMAGE_NAME}:${VERSION}${NC}"
echo ""

# Check if logged in to Docker Hub
echo -e "${YELLOW}Checking Docker Hub login...${NC}"
if ! docker info 2>/dev/null | grep -q "Username: ${DOCKER_USERNAME}"; then
    echo -e "${YELLOW}Please login to Docker Hub:${NC}"
    docker login -u "${DOCKER_USERNAME}"
fi

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

echo -e "${YELLOW}Building from: ${PROJECT_ROOT}${NC}"

# Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    -f deployment/docker/Dockerfile.production \
    -t "${FULL_IMAGE_NAME}:${VERSION}" \
    -t "${FULL_IMAGE_NAME}:latest" \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker build successful${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Show image size
echo -e "${YELLOW}Image size:${NC}"
docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

# Push to Docker Hub
echo -e "${GREEN}Pushing to Docker Hub...${NC}"
docker push "${FULL_IMAGE_NAME}:${VERSION}"
docker push "${FULL_IMAGE_NAME}:latest"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully pushed to Docker Hub${NC}"
    echo -e "${GREEN}Image available at: https://hub.docker.com/r/${FULL_IMAGE_NAME}${NC}"
else
    echo -e "${RED}✗ Docker push failed${NC}"
    exit 1
fi

# Test the image locally
echo -e "${YELLOW}Testing image locally...${NC}"
docker run -d \
    --name ai-agent-test \
    -p 8000:8000 \
    -e CHROMA_HOST=43.205.192.16 \
    -e CHROMA_PORT=8000 \
    "${FULL_IMAGE_NAME}:${VERSION}"

sleep 5

# Check if container is running
if docker ps | grep -q ai-agent-test; then
    echo -e "${GREEN}✓ Container started successfully${NC}"
    
    # Test health endpoint
    if curl -f http://localhost:8000/health 2>/dev/null; then
        echo -e "${GREEN}✓ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠ Health check failed (API might still be starting)${NC}"
    fi
    
    # Cleanup test container
    docker stop ai-agent-test
    docker rm ai-agent-test
else
    echo -e "${RED}✗ Container failed to start${NC}"
    docker logs ai-agent-test
    docker rm ai-agent-test
    exit 1
fi

echo ""
echo -e "${GREEN}=== Build & Push Complete ===${NC}"
echo -e "Image: ${YELLOW}${FULL_IMAGE_NAME}:${VERSION}${NC}"
echo -e "Also tagged as: ${YELLOW}${FULL_IMAGE_NAME}:latest${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Update kubernetes deployment with new image:"
echo "   ${YELLOW}${FULL_IMAGE_NAME}:${VERSION}${NC}"
echo "2. Deploy to Kubernetes:"
echo "   ${YELLOW}./deployment/scripts/deploy.sh${NC}"