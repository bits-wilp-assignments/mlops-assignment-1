# Heart Disease API Helm Chart

Helm chart for deploying the Heart Disease Prediction API on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- NGINX Ingress Controller installed in cluster

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
| `service.type` | Service type | `ClusterIP` |
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress controller class name | `nginx` |
| `ingress.host` | Ingress hostname | `heart-disease-api.local` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |
| `autoscaling.enabled` | Enable HPA | `false` |
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

**Setup DNS/Hosts Entry:**

For local testing, add to `/etc/hosts`:

```bash
echo "127.0.0.1 heart-disease-api.local" | sudo tee -a /etc/hosts
```

**For Minikube:**

```bash
minikube tunnel
```

**Get Ingress details:**

```bash
kubectl get ingress
```

**Access via Ingress:**

```bash
# Access at: http://heart-disease-api.local
# Swagger UI: http://heart-disease-api.local/apidocs
```

**Alternative - Port forward for direct access:**

```bash
kubectl port-forward service/heart-disease-api 5555:5555

# Access at: http://localhost:5555
```

## Test the Deployment

```bash
# Health check
curl http://heart-disease-api.local/health

# Make prediction
curl -X POST http://heart-disease-api.local/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
       "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
       "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```
```

## Monitoring

**Check deployment status:**

```bash
kubectl get pods
kubectl get hpa
kubectl logs -f <pod-name>
```
