# GR00T Inference Service - RunPod Deployment

This document provides instructions for deploying the GR00T inference service on RunPod using the automatically built Docker image.

## Docker Image

The Docker image is automatically built and pushed to GitHub Container Registry (GHCR) on every push to main branch:

```
ghcr.io/your-username/isaac-gr00t:main
```

## RunPod Template Configuration

### Container Configuration

- **Container Image**: `ghcr.io/your-username/isaac-gr00t:main`
- **Container Start Command**: `/usr/bin/python3 /workspace/scripts/inference_service.py`

### Environment Variables

Configure the following environment variables in your RunPod template:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `nvidia/GR00T-N1-2B` | Local path or HuggingFace repo ID |
| `EMBODIMENT_TAG` | `None` (auto-detect) | Embodiment tag from metadata.json |
| `NUM_ARMS` | `1` | Number of robot arms |
| `NUM_CAMS` | `2` | Number of cameras |
| `DENOISING_STEPS` | `4` | Diffusion denoising steps |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `5555` | Server port |

### Example Environment Variables

For a typical single-arm Franka Panda setup:
```
MODEL_PATH=nvidia/GR00T-N1-2B
EMBODIMENT_TAG=franka_panda
NUM_ARMS=1
NUM_CAMS=2
DENOISING_STEPS=4
HOST=0.0.0.0
PORT=5555
```

### Exposed Ports

Configure RunPod to expose port `5555` (or your custom PORT value) for the inference service.

## Container Start Command with Environment Variables

The inference service will automatically use environment variables if they are set. The start command becomes:

```bash
python3 /workspace/scripts/inference_service.py \
  --model_path ${MODEL_PATH:-nvidia/GR00T-N1-2B} \
  --embodiment_tag ${EMBODIMENT_TAG} \
  --num_arms ${NUM_ARMS:-1} \
  --num_cams ${NUM_CAMS:-2} \
  --denoising_steps ${DENOISING_STEPS:-4} \
  --host ${HOST:-0.0.0.0} \
  --port ${PORT:-5555}
```

## Volume Mounts (Optional)

If you want to use local models, mount them to `/workspace/models`:

- **Container Path**: `/workspace/models`
- **Volume Size**: Depends on model size (typically 10-50GB)

Then set `MODEL_PATH=/workspace/models/your-model-name`

## GPU Requirements

- **GPU Type**: NVIDIA GPU with CUDA support
- **VRAM**: Minimum 8GB, recommended 16GB+ for larger models
- **GPU Count**: 1 (single GPU inference)

## Network Configuration

- **HTTP Port**: 5555 (configurable via PORT env var)
- **Protocol**: HTTP REST API
- **Health Check**: GET `/health` (if implemented)

## Auto-scaling

The service supports horizontal scaling by running multiple instances behind a load balancer, as each instance is stateless.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase GPU VRAM or reduce model size
2. **Model Loading Fails**: Check MODEL_PATH and network connectivity
3. **Embodiment Tag Issues**: Let it auto-detect or check metadata.json

### Logs

Check RunPod logs for:
- Model loading progress
- Auto-detected embodiment tag
- Server startup confirmation
- Any error messages

### Health Check

The service should log:
```
Loaded Gr00tPolicy from nvidia/GR00T-N1-2B [embodiment=franka_panda] with 4 steps.
Auto-detected embodiment tag: franka_panda
Starting Gr00t server at 0.0.0.0:5555
```

## API Usage

Once deployed, you can send inference requests to:
```
http://your-runpod-endpoint:5555/
```

Refer to the GR00T documentation for API endpoints and request formats.
