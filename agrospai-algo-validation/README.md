# Commands

## Build the image

```bash
docker buildx build --platform linux/amd64 -t appleseg-validated . --load
```

## Tag and push the image

Make sure to use the `linux/amd64` platform when building the image.

```bash
docker tag appleseg-validated registry.agrospai.udl.cat/library/appleseg-validated:latest
docker push registry.agrospai.udl.cat/library/appleseg-validated:latest
```

## Run the container

```bash
docker run --rm -it \
    --name appleseg_validator \
    -v ./data:/data \
    -e DEV=1 \
    -e DIDS='["eb60f87363a36a5ae5cb8373524a8fd976b0cc5f8c40a706c615b857ae0e2974"]' \
    appleseg-validated
```

## Debug mode

```bash
docker run --rm -it \
    --name appleseg_validator \
    -v ./data:/data \
    -e DEV=1 \
    -e DIDS='["eb60f87363a36a5ae5cb8373524a8fd976b0cc5f8c40a706c615b857ae0e2974"]' \
    appleseg-validated tail -f /dev/null
```

```bash
docker exec -it appleseg_validator /bin/bash
```
