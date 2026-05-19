# Пайплайн по распознаванию ценников по видео

Production-ready микросервисное решение для автоматического распознавания, трекинга и парсинга ценников с видеопотока. Проект разработан для автоматизации аудита полок в ритейле.

## Архитектура решения

Наш проект построен на базе **асинхронной микросервисной архитектуры**, что позволяет горизонтально масштабировать систему и обрабатывать тяжелые видеофайлы без блокировки пользовательских интерфейсов.

### Инфраструктурный слой:
*   **FastAPI** — асинхронный веб-сервер, выступающий точкой входа (Gateway). Моментально принимает видео, сохраняет их и отдает клиенту `task_id`, не дожидаясь окончания ML-анализа.
*   **Celery** — фоновый распределенный обработчик задач. Запускает тяжелый ML-пайплайн.
*   **Redis** — in-memory брокер сообщений (Message Broker) для связи FastAPI и Celery.
*   **MongoDB** — NoSQL база данных для хранения состояний и прогресса обработки видео-задач.
*   **Docker Compose** — оркестратор контейнеров, обеспечивающий изоляцию зависимостей и платформонезависимость.

### ML-Пайплайн:
1.  **Object Tracking (YOLO):** Мы отказались от покадровой независимой детекции. Модель трекает физические ценники в пространстве, присваивая им уникальные ID.
2.  **Temporal Sampling (Оптимизация):** Для каждого `object_id` извлекается лишь каждый 6-й кадр (Stride=6). Это снижает вычислительную нагрузку на CPU в 6 раз без потери точности.
3.  **ONNX Runtime OCR (RapidOCR):** Использование легковесных ONNX-моделей вместо тяжелых фреймворков. Обеспечивает стабильный и быстрый инференс на CPU без конфликтов С++ библиотек.
4.  **Spatial Clustering & NLP Parsing:** Эвристический парсер, устойчивый к бликам и сложному освещению магазина.

---

# Быстрый старт

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
