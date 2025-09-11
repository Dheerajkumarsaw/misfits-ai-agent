#!/bin/bash

# Setup Kubernetes cluster on EC2 instances
# Run this on each EC2 instance (master and workers)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect if this is master or worker
IS_MASTER=false
if [ "$1" == "master" ]; then
    IS_MASTER=true
fi

echo -e "${BLUE}=== Kubernetes Cluster Setup on EC2 ===${NC}"
echo -e "Mode: ${YELLOW}$([ "$IS_MASTER" = true ] && echo "MASTER" || echo "WORKER")${NC}"
echo ""

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo -e "${BLUE}Installing Docker...${NC}"
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Configure Docker
sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# Add current user to docker group
sudo usermod -aG docker $USER

echo -e "${GREEN}✓ Docker installed${NC}"

# Disable swap
echo -e "${YELLOW}Disabling swap...${NC}"
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Install Kubernetes components
echo -e "${BLUE}Installing Kubernetes components...${NC}"
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet=1.28.0-00 kubeadm=1.28.0-00 kubectl=1.28.0-00
sudo apt-mark hold kubelet kubeadm kubectl

# Enable kernel modules
sudo modprobe br_netfilter
sudo modprobe overlay

# Configure sysctl
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system

echo -e "${GREEN}✓ Kubernetes components installed${NC}"

if [ "$IS_MASTER" = true ]; then
    echo ""
    echo -e "${BLUE}=== Master Node Setup ===${NC}"
    
    # Get private IP
    PRIVATE_IP=$(hostname -I | awk '{print $1}')
    
    # Initialize cluster
    echo -e "${YELLOW}Initializing Kubernetes cluster...${NC}"
    sudo kubeadm init \
        --apiserver-advertise-address=$PRIVATE_IP \
        --pod-network-cidr=10.244.0.0/16 \
        --ignore-preflight-errors=NumCPU
    
    # Configure kubectl
    echo -e "${YELLOW}Configuring kubectl...${NC}"
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    
    # Install Flannel network plugin
    echo -e "${YELLOW}Installing Flannel network plugin...${NC}"
    kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
    
    # Install metrics server
    echo -e "${YELLOW}Installing metrics server...${NC}"
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Patch metrics server for insecure TLS
    kubectl patch deployment metrics-server -n kube-system --type='json' \
        -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
    
    # Generate join command
    echo ""
    echo -e "${GREEN}=== Cluster Initialized Successfully ===${NC}"
    echo ""
    echo -e "${YELLOW}Save this join command for worker nodes:${NC}"
    echo -e "${BLUE}---${NC}"
    kubeadm token create --print-join-command
    echo -e "${BLUE}---${NC}"
    
    # Wait for node to be ready
    echo ""
    echo -e "${YELLOW}Waiting for master node to be ready...${NC}"
    kubectl wait --for=condition=Ready node --all --timeout=300s
    
    # Show cluster info
    echo ""
    echo -e "${GREEN}Cluster Status:${NC}"
    kubectl get nodes
    kubectl get pods --all-namespaces
    
else
    echo ""
    echo -e "${BLUE}=== Worker Node Setup Complete ===${NC}"
    echo ""
    echo -e "${YELLOW}To join this node to the cluster, run the join command from the master node:${NC}"
    echo "sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>"
fi

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"

if [ "$IS_MASTER" = true ]; then
    echo -e "Master node is ready. Deploy your application with:"
    echo -e "  ${YELLOW}./deployment/scripts/deploy.sh${NC}"
else
    echo -e "Worker node is ready. Join it to the cluster using the command from master."
fi