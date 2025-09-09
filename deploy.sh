#!/bin/bash

# Meetup Recommendation API Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Meetup Recommendation API Deployment Script${NC}"
echo "=============================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Configuration
DOCKER_IMAGE="dsaw1/misfits-event-recommendation"
TAG=${2:-latest}

# Function to build Docker image
build_image() {
    echo -e "${BLUE}📦 Building Docker image...${NC}"
    docker build -t ${DOCKER_IMAGE}:${TAG} .
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
}

# Function to push Docker image to Docker Hub
push_image() {
    echo -e "${BLUE}🚀 Pushing Docker image to Docker Hub...${NC}"
    
    # Check if logged in to Docker Hub
    if ! docker info | grep -q "Username"; then
        echo -e "${YELLOW}⚠️  Please login to Docker Hub first:${NC}"
        echo "docker login"
        exit 1
    fi
    
    echo "Pushing ${DOCKER_IMAGE}:${TAG}..."
    docker push ${DOCKER_IMAGE}:${TAG}
    echo -e "${GREEN}✅ Docker image pushed successfully${NC}"
}

# Function to run with Docker Compose
run_compose() {
    echo -e "${BLUE}🐳 Starting with Docker Compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Application started with Docker Compose${NC}"
    echo -e "${YELLOW}🌐 API will be available at: http://localhost:8000${NC}"
    echo -e "${YELLOW}📊 Health check: http://localhost:8000/health/liveness${NC}"
}

# Function to deploy to Kubernetes
deploy_k8s() {
    echo -e "${BLUE}☸️  Deploying to Kubernetes...${NC}"
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        exit 1
    fi
    
    # Push image to Docker Hub first
    push_image
    
    # Apply ConfigMap and Secrets first
    echo "📝 Applying ConfigMap and Secrets..."
    kubectl apply -f k8s-configmap.yaml
    
    # Apply Deployment and Service
    echo "🚀 Applying Deployment and Service..."
    kubectl apply -f k8s-deployment.yaml
    
    # Apply Ingress (optional)
    if [ -f "k8s-ingress.yaml" ]; then
        echo "🌐 Applying Ingress..."
        kubectl apply -f k8s-ingress.yaml
    fi
    
    echo -e "${GREEN}✅ Kubernetes deployment completed${NC}"
    echo "📊 Check deployment status:"
    echo "   kubectl get pods -l app=misfits-event-recommendation"
    echo "   kubectl get services"
}

# Function to stop services
stop_services() {
    echo -e "${BLUE}🛑 Stopping services...${NC}"
    
    if [ "$1" = "k8s" ]; then
        echo "☸️  Stopping Kubernetes deployment..."
        kubectl delete -f k8s-deployment.yaml || true
        kubectl delete -f k8s-configmap.yaml || true
        kubectl delete -f k8s-ingress.yaml || true
        echo -e "${GREEN}✅ Kubernetes resources deleted${NC}"
    else
        echo "🐳 Stopping Docker Compose..."
        docker-compose down
        echo -e "${GREEN}✅ Docker Compose stopped${NC}"
    fi
}

# Function to show logs
show_logs() {
    if [ "$1" = "k8s" ]; then
        echo "📋 Kubernetes logs:"
        kubectl logs -l app=meetup-api --tail=50
    else
        echo "📋 Docker Compose logs:"
        docker-compose logs -f
    fi
}

# Main menu
case "${1:-help}" in
    "build")
        build_image
        ;;
    "push")
        push_image
        ;;
    "build-push")
        build_image
        push_image
        ;;
    "run"|"start")
        build_image
        run_compose
        ;;
    "k8s"|"kubernetes")
        build_image
        deploy_k8s
        ;;
    "stop")
        stop_services "${2:-docker}"
        ;;
    "logs")
        show_logs "${2:-docker}"
        ;;
    "status")
        echo -e "${BLUE}📊 Service Status:${NC}"
        echo "🐳 Docker Compose:"
        docker-compose ps || echo "Not running"
        echo ""
        echo "☸️  Kubernetes:"
        kubectl get pods -l app=misfits-event-recommendation 2>/dev/null || echo "Not deployed or kubectl not available"
        ;;
    "help"|*)
        echo -e "${BLUE}Usage: ./deploy.sh [COMMAND] [TAG]${NC}"
        echo ""
        echo "Commands:"
        echo "  build         - Build Docker image only"
        echo "  push          - Push Docker image to Docker Hub"
        echo "  build-push    - Build and push Docker image"
        echo "  run           - Build and run with Docker Compose"
        echo "  k8s           - Build, push and deploy to Kubernetes"
        echo "  stop          - Stop services (add 'k8s' for Kubernetes)"
        echo "  logs          - Show logs (add 'k8s' for Kubernetes)"
        echo "  status        - Show service status"
        echo "  help          - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh build-push           - Build and push with 'latest' tag"
        echo "  ./deploy.sh build-push v1.0.0    - Build and push with 'v1.0.0' tag"
        echo "  ./deploy.sh k8s                  - Full deployment to Kubernetes"
        echo "  ./deploy.sh run                  - Start with Docker Compose"
        echo "  ./deploy.sh stop k8s             - Stop Kubernetes deployment"
        ;;
esac