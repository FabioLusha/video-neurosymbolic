
Build with:
```bash
docker build \
  --build-arg USERNAME="$USER" \
  --build-arg USER_UID="$(id -u)" \
  -t myimage:latest .
```
