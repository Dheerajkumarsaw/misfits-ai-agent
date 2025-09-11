#!/bin/bash

# Deploy AI Agent to Kubernetes Cluster
# This script should be run on the Kubernetes master node

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== AI Agent Kubernetes Deployment ===${NC}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check cluster connection
echo -e "${YELLOW}Checking Kubernetes cluster connection...${NC}"
if kubectl cluster-info &> /dev/null; then
    echo -e "${GREEN}✓ Connected to Kubernetes cluster${NC}"
    kubectl cluster-info | head -n 1
else
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    echo "Please ensure kubectl is configured correctly"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="${SCRIPT_DIR}/../k8s"

# Check if k8s directory exists
if [ ! -d "$K8S_DIR" ]; then
    echo -e "${RED}Kubernetes manifests directory not found: $K8S_DIR${NC}"
    exit 1
fi

# Function to apply manifests
apply_manifest() {
    local file=$1
    local description=$2
    
    echo -e "${YELLOW}Applying ${description}...${NC}"
    if kubectl apply -f "$file"; then
        echo -e "${GREEN}✓ ${description} applied${NC}"
    else
        echo -e "${RED}✗ Failed to apply ${description}${NC}"
        return 1
    fi
}

# Create namespace first
echo -e "${BLUE}Step 1: Creating namespace${NC}"
apply_manifest "$K8S_DIR/00-namespace.yaml" "Namespace"
echo ""

# Wait for namespace to be ready
sleep 2

# Apply configurations in order
echo -e "${BLUE}Step 2: Applying configurations${NC}"

# Check if secrets file has been updated with actual API key
if grep -q "YOUR-ACTUAL-API-KEY-HERE" "$K8S_DIR/01-secrets.yaml"; then
    echo -e "${YELLOW}⚠ WARNING: Secrets file contains placeholder API key${NC}"
    echo -e "${YELLOW}Please update $K8S_DIR/01-secrets.yaml with your actual NVIDIA API key${NC}"
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment cancelled${NC}"
        exit 1
    fi
fi

apply_manifest "$K8S_DIR/01-secrets.yaml" "Secrets"
apply_manifest "$K8S_DIR/02-configmap.yaml" "ConfigMap"
echo ""

# Check if deployment has correct Docker image
if grep -q "YOUR_DOCKERHUB_USERNAME" "$K8S_DIR/03-deployment.yaml"; then
    echo -e "${RED}ERROR: Deployment file contains placeholder Docker Hub username${NC}"
    echo -e "${YELLOW}Please update $K8S_DIR/03-deployment.yaml with your Docker Hub username${NC}"
    echo -e "${YELLOW}Replace 'YOUR_DOCKERHUB_USERNAME/ai-agent:latest' with your actual image${NC}"
    exit 1
fi

echo -e "${BLUE}Step 3: Deploying application${NC}"
apply_manifest "$K8S_DIR/03-deployment.yaml" "Deployment"
echo ""

echo -e "${BLUE}Step 4: Creating services${NC}"
apply_manifest "$K8S_DIR/04-service.yaml" "Services"
echo ""

echo -e "${BLUE}Step 5: Setting up ingress (optional)${NC}"
if kubectl get ingressclass 2>/dev/null | grep -q nginx; then
    apply_manifest "$K8S_DIR/05-ingress.yaml" "Ingress"
else
    echo -e "${YELLOW}⚠ No ingress controller found, skipping ingress setup${NC}"
fi
echo ""

echo -e "${BLUE}Step 6: Configuring auto-scaling${NC}"
apply_manifest "$K8S_DIR/06-hpa.yaml" "HorizontalPodAutoscaler"
echo ""

# Wait for deployment
echo -e "${BLUE}Step 7: Waiting for deployment to be ready${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

kubectl rollout status deployment/ai-agent -n ai-agent-prod --timeout=300s

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Deployment successful${NC}"
else
    echo -e "${RED}✗ Deployment failed or timed out${NC}"
    echo -e "${YELLOW}Checking pod status...${NC}"
    kubectl get pods -n ai-agent-prod
    echo ""
    echo -e "${YELLOW}Recent events:${NC}"
    kubectl get events -n ai-agent-prod --sort-by='.lastTimestamp' | tail -10
    exit 1
fi

echo ""
echo -e "${BLUE}=== Deployment Status ===${NC}"
kubectl get all -n ai-agent-prod
echo ""

# Get service information
echo -e "${BLUE}=== Access Information ===${NC}"

# Get NodePort
NODE_PORT=$(kubectl get svc ai-agent-service -n ai-agent-prod -o jsonpath='{.spec.ports[0].nodePort}')
echo -e "${GREEN}NodePort Service:${NC}"
echo -e "  Port: ${YELLOW}${NODE_PORT}${NC}"

# Get node IPs
echo -e "${GREEN}Node IPs:${NC}"
kubectl get nodes -o wide | awk '{print $6}' | tail -n +2 | while read ip; do
    echo -e "  ${YELLOW}http://${ip}:${NODE_PORT}${NC}"
done

# Check if LoadBalancer is available
LB_IP=$(kubectl get svc ai-agent-lb -n ai-agent-prod -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
if [ ! -z "$LB_IP" ]; then
    echo -e "${GREEN}LoadBalancer URL:${NC}"
    echo -e "  ${YELLOW}http://${LB_IP}${NC}"
fi

echo ""
echo -e "${BLUE}=== Quick Commands ===${NC}"
echo -e "${GREEN}View logs:${NC}"
echo "  kubectl logs -f deployment/ai-agent -n ai-agent-prod"
echo ""
echo -e "${GREEN}Scale deployment:${NC}"
echo "  kubectl scale deployment ai-agent --replicas=5 -n ai-agent-prod"
echo ""
echo -e "${GREEN}Port forward for local testing:${NC}"
echo "  kubectl port-forward service/ai-agent-service 8000:8000 -n ai-agent-prod"
echo ""
echo -e "${GREEN}Check pod status:${NC}"
echo "  kubectl get pods -n ai-agent-prod"
echo ""
echo -e "${GREEN}Describe deployment:${NC}"
echo "  kubectl describe deployment ai-agent -n ai-agent-prod"
echo ""

# Test the deployment
echo -e "${BLUE}=== Testing Deployment ===${NC}"
echo -e "${YELLOW}Attempting to test health endpoint...${NC}"

# Try port-forward in background for testing
kubectl port-forward service/ai-agent-service 8888:8000 -n ai-agent-prod &> /dev/null &
PF_PID=$!
sleep 5

if curl -f http://localhost:8888/health 2>/dev/null; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${YELLOW}⚠ Could not test health endpoint automatically${NC}"
    echo -e "${YELLOW}Please test manually using one of the URLs above${NC}"
fi

# Kill port-forward
kill $PF_PID 2>/dev/null

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo -e "Your AI Agent is now running on Kubernetes!"