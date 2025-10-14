# Docker-in-Docker (DinD) setup for AgroSpAI algorithm validation

```bash
docker buildx build --platform linux/amd64 -t registry.agrospai.udl.cat/library/algo-validation-dind:latest -f Dockerfile . --push
```
