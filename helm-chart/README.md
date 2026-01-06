# Heart Disease API Helm Chart

A Helm chart for deploying the Heart Disease Prediction API on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installing the Chart

To install the chart with the release name `heart-disease-api`:

```bash
helm install heart-disease-api ./helm-chart
```

## Uninstalling the Chart

To uninstall/delete the `heart-disease-api` deployment:

```bash
helm uninstall heart-disease-api
```

## Configuration

The following table lists the configurable parameters of the Heart Disease API chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `2` |
| `image.repository` | Image repository | `bitshub4krishanu/heart-disease-api` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Kubernetes service type | `NodePort` |
| `service.port` | Service port | `5555` |
| `service.nodePort` | NodePort (if service type is NodePort) | `30555` |
| `env` | Environment variables | See `values.yaml` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `512Mi` |

## Accessing the API

Once deployed, the API will be accessible at:

```
http://<NODE_IP>:30555
```

To get the node IP:

```bash
kubectl get nodes -o wide
```

### Test the API

```bash
# Health check
curl http://<NODE_IP>:30555/health

# Model info
curl http://<NODE_IP>:30555/model-info

# Prediction
curl -X POST http://<NODE_IP>:30555/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
      }
    ]
  }'
```

## Customizing the Installation

You can override values by creating a custom `values.yaml` file or using `--set`:

```bash
# Using custom values file
helm install heart-disease-api ./helm-chart -f custom-values.yaml

# Using --set
helm install heart-disease-api ./helm-chart \
  --set replicaCount=3 \
  --set image.tag=v1.0.0 \
  --set env[0].value="models:/heart_disease_best_model@staging"
```

## Upgrading

To upgrade an existing release:

```bash
helm upgrade heart-disease-api ./helm-chart
```

## Autoscaling

To enable horizontal pod autoscaling:

```bash
helm install heart-disease-api ./helm-chart \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10 \
  --set autoscaling.targetCPUUtilizationPercentage=80
```
