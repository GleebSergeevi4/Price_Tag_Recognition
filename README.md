# Quick Start

## 1) Requirements

- Docker Desktop (with Docker Compose)

## 2) Build Images

```bash
docker compose build
```

## 3) Start Services

```bash
docker compose up -d --force-recreate
```

## 4) Check Status

```bash
docker compose ps
```

## 5) Check Logs

```bash
docker compose logs -f api
docker compose logs -f worker
```

## 6) Stop Services

```bash
docker compose down
```

## 7) Full Reset (optional)

```bash
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

## 8) Useful Endpoints

- API: http://localhost:8000
- Mongo Express (dev profile only): http://localhost:8081

To start with dev profile:

```bash
docker compose --profile dev up -d
```
