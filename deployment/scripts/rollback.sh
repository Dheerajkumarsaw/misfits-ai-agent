#!/bin/bash

# Rollback AI Agent deployment to previous version

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="ai-agent-prod"
DEPLOYMENT="ai-agent"

echo -e "${BLUE}=== AI Agent Deployment Rollback ===${NC}"
echo ""

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed${NC}"
    exit 1
fi

# Show rollout history
echo -e "${YELLOW}Rollout history:${NC}"
kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE

# Get current status
echo ""
echo -e "${YELLOW}Current deployment status:${NC}"
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Ask for confirmation
echo ""
read -p "Do you want to rollback to the previous version? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Rollback cancelled${NC}"
    exit 1
fi

# Perform rollback
echo -e "${YELLOW}Rolling back deployment...${NC}"
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback
echo -e "${YELLOW}Waiting for rollback to complete...${NC}"
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Rollback successful${NC}"
    
    # Show new status
    echo ""
    echo -e "${BLUE}New deployment status:${NC}"
    kubectl get deployment $DEPLOYMENT -n $NAMESPACE
    kubectl get pods -n $NAMESPACE -l app=ai-agent
else
    echo -e "${RED}✗ Rollback failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Rollback Complete ===${NC}"