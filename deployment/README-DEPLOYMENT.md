# AI Agent Production Deployment Guide

This guide will help you deploy your AI Agent to a Kubernetes cluster running on EC2 instances.

## Prerequisites

- Docker Hub account
- EC2 instances running (1 master, 2+ workers)
- SSH access to EC2 instances
- kubectl configured (on master node)
- Your NVIDIA API key

## Quick Start

### 1. Build and Push Docker Image (Run Locally)

```bash
# Make scripts executable
chmod +x deployment/scripts/*.sh

# Build and push to Docker Hub
./deployment/scripts/build-push.sh YOUR_DOCKERHUB_USERNAME v1.0.0
```

### 2. Prepare Kubernetes Manifests

Before deploying, update these files:

#### Update Docker Image
Edit `deployment/k8s/03-deployment.yaml`:
```yaml
# Line 24: Replace with your Docker Hub username
image: YOUR_DOCKERHUB_USERNAME/ai-agent:latest
```

#### Update API Key
Edit `deployment/k8s/01-secrets.yaml`:
```yaml
# Line 10: Replace with your actual NVIDIA API key
OPENAI_API_KEY: "nvapi-YOUR-ACTUAL-API-KEY-HERE"
```

### 3. Deploy to Kubernetes (Run on Master Node)

```bash
# Copy deployment files to master node
scp -r deployment/ ubuntu@<master-ip>:~/

# SSH to master node
ssh ubuntu@<master-ip>

# Make scripts executable
chmod +x deployment/scripts/*.sh

# Deploy application
cd deployment
./scripts/deploy.sh
```

## Detailed Setup Guide

### Step 1: Setup Kubernetes Cluster (If Not Already Done)

If you haven't set up Kubernetes on your EC2 instances yet:

#### On Master Node:
```bash
# Run setup script
./deployment/scripts/setup-k8s-cluster.sh master

# Save the join command output for worker nodes
```

#### On Each Worker Node:
```bash
# Run setup script
./deployment/scripts/setup-k8s-cluster.sh worker

# Then run the join command from master
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

### Step 2: Build Docker Image

```bash
# From project root on your local machine
cd /Users/mac/Downloads/Ai\ Agents

# Build and push (replace with your Docker Hub username)
./deployment/scripts/build-push.sh YOUR_USERNAME v1.0.0
```

This will:
- Build optimized production image
- Push to Docker Hub
- Test the image locally
- Tag as both version and latest

### Step 3: Configure Kubernetes Resources

All configuration files are in `deployment/k8s/`:

- `00-namespace.yaml` - Isolated namespace
- `01-secrets.yaml` - API keys (UPDATE REQUIRED)
- `02-configmap.yaml` - Environment variables
- `03-deployment.yaml` - Main application (UPDATE REQUIRED)
- `04-service.yaml` - Service exposure
- `05-ingress.yaml` - Optional HTTPS routing
- `06-hpa.yaml` - Auto-scaling

### Step 4: Deploy Application

```bash
# On master node
cd deployment
./scripts/deploy.sh
```

The script will:
1. Create namespace
2. Apply secrets and configmap
3. Deploy application
4. Create services
5. Setup auto-scaling
6. Show access URLs

### Step 5: Access Your Application

After deployment, access your application via:

#### NodePort (Default)
```bash
http://<any-node-ip>:30080
```

#### Port Forwarding (For Testing)
```bash
kubectl port-forward service/ai-agent-service 8000:8000 -n ai-agent-prod
# Access at http://localhost:8000
```

#### Load Balancer (If Configured)
```bash
http://<load-balancer-url>
```

## Common Operations

### View Logs
```bash
kubectl logs -f deployment/ai-agent -n ai-agent-prod
```

### Scale Deployment
```bash
kubectl scale deployment ai-agent --replicas=5 -n ai-agent-prod
```

### Update Image
```bash
# Build and push new version
./deployment/scripts/build-push.sh YOUR_USERNAME v1.0.1

# Update deployment
kubectl set image deployment/ai-agent ai-agent=YOUR_USERNAME/ai-agent:v1.0.1 -n ai-agent-prod
```

### Rollback
```bash
./deployment/scripts/rollback.sh
```

### Check Status
```bash
kubectl get all -n ai-agent-prod
kubectl get pods -n ai-agent-prod
kubectl describe deployment ai-agent -n ai-agent-prod
```

## Testing the API

### Health Check
```bash
curl http://<node-ip>:30080/health
```

### Chat Endpoint
```bash
curl -X POST http://<node-ip>:30080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "find tech events",
    "user_id": "user123",
    "user_current_city": "Mumbai"
  }'
```

### WebSocket Connection
```bash
wscat -c ws://<node-ip>:30080/ws/chat
```

## Monitoring

### View Metrics
```bash
kubectl top nodes
kubectl top pods -n ai-agent-prod
```

### Check HPA Status
```bash
kubectl get hpa -n ai-agent-prod
```

### View Events
```bash
kubectl get events -n ai-agent-prod --sort-by='.lastTimestamp'
```

## Troubleshooting

### Pod Not Starting
```bash
# Check pod status
kubectl get pods -n ai-agent-prod

# View pod logs
kubectl logs <pod-name> -n ai-agent-prod

# Describe pod for events
kubectl describe pod <pod-name> -n ai-agent-prod
```

### Image Pull Error
- Verify Docker Hub credentials
- Check image name in deployment.yaml
- Ensure image exists on Docker Hub

### API Not Responding
- Check ChromaDB connectivity (43.205.192.16:8000)
- Verify NVIDIA API key in secrets
- Check pod logs for errors

### Out of Memory
- Increase resource limits in deployment.yaml
- Scale horizontally (more replicas)
- Check for memory leaks in logs

## Security Best Practices

1. **Never commit secrets** - Use Kubernetes secrets
2. **Use RBAC** - Limit access to namespace
3. **Network Policies** - Restrict pod communication
4. **Update regularly** - Keep images and K8s updated
5. **Monitor logs** - Watch for suspicious activity

## Cost Optimization

- Use t3.medium instances (~$30/month each)
- Enable auto-scaling (2-10 replicas)
- Use spot instances for workers
- Monitor and optimize resource requests/limits

## Next Steps

1. Setup domain and SSL certificate
2. Configure monitoring (Prometheus/Grafana)
3. Setup CI/CD pipeline
4. Implement backup strategy
5. Add authentication to API

## Support

For issues or questions:
- Check pod logs first
- Review events in namespace
- Verify all configurations
- Test connectivity to ChromaDB

## File Structure

```
deployment/
├── docker/
│   ├── Dockerfile.production
│   └── .dockerignore
├── k8s/
│   ├── 00-namespace.yaml
│   ├── 01-secrets.yaml
│   ├── 02-configmap.yaml
│   ├── 03-deployment.yaml
│   ├── 04-service.yaml
│   ├── 05-ingress.yaml
│   └── 06-hpa.yaml
├── scripts/
│   ├── build-push.sh
│   ├── deploy.sh
│   ├── rollback.sh
│   └── setup-k8s-cluster.sh
└── README-DEPLOYMENT.md
```