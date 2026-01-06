# Heart Disease API Helm Chart

Helm chart for deploying the Heart Disease Prediction API on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Quick Start

**Install:**

```bash
helm install heart-disease-api ./helm-chart
```

**Upgrade:**

```bash
helm upgrade heart-disease-api ./helm-chart
```

**Uninstall:**

```bash
helm uninstall heart-disease-api
```

## Configuration

Key configurable parameters in [values.yaml](values.yaml):

| Parameter | Description | Default |
| --- | --- | --- |
| `replicaCount` | Number of replicas | `2` |
| `image.repository` | Docker image repository | `bitshub4krishanu/heart-disease-api` |
| `image.tag` | Image tag | `latest` |
| `service.type` | Service type (NodePort/ClusterIP/LoadBalancer) | `NodePort` |
| `service.nodePort` | NodePort for external access | `30555` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.maxReplicas` | Maximum replicas | `5` |
| `env` | Environment variables (MODEL_URI, etc.) | See values.yaml |

## Customize Installation

**Using custom values file:**

```bash
helm install heart-disease-api ./helm-chart -f custom-values.yaml
```

**Using --set flags:**

```bash
helm install heart-disease-api ./helm-chart \
  --set replicaCount=3 \
  --set image.tag=v1.0.0
```

## Access the API

**Get service details:**

```bash
kubectl get svc heart-disease-api
```

**For NodePort (default):**

```bash
# Get node IP
kubectl get nodes -o wide

# Access at: http://<NODE_IP>:30555
```

**Port forward for local testing:**

```bash
kubectl port-forward service/heart-disease-api 5555:5555

# Access at: http://localhost:5555
# Swagger UI: http://localhost:5555/apidocs
```

## Test the Deployment

```bash
# Health check
curl http://localhost:5555/health

# Make prediction
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
       "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
       "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

## Monitoring

**Check deployment status:**

```bash
kubectl get pods
kubectl get hpa
kubectl logs -f <pod-name>
```
