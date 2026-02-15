# ML Inference API (Dockerized FastAPI Service)

## 1. Project Overview

This project implements a production-style machine learning inference service using:

- Scikit-learn (Logistic Regression)
- FastAPI
- Docker
- Pytest

The service exposes a REST endpoint for predictions and runs inside a fully containerized environment.

---

## 🎯 Why This Matters

This project demonstrates **production-ready ML architecture**:

- **Separation of Concerns:** Training logic isolated from serving logic
- **Reproducibility:** Docker ensures consistent environments across dev/staging/production
- **Scalability:** REST API can handle concurrent requests with Uvicorn workers
- **Type Safety:** Pydantic validates all inputs; breaking changes are caught early
- **Testing:** Comprehensive pytest coverage ensures reliability
- **Self-Healing:** Container automatically trains models if artifacts don't exist

Real-world ML services follow this exact pattern. This portfolio piece demonstrates you understand enterprise ML deployment.

## 2. Architecture

```
Client Request (JSON)
        ↓
FastAPI Endpoint
        ↓
Pydantic Validation
        ↓
Preprocessing (StandardScaler)
        ↓
Trained Logistic Regression Model
        ↓
JSON Response (prediction + probability)
```

## 3. Tech Stack

- Python 3.11
- Scikit-learn
- FastAPI
- Uvicorn
- Pytest
- Docker

## 4. Model Details

- **Dataset:** Iris
- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler
- **Train/Test Split:** 80/20
- **Model artifacts saved with:** joblib

## 5. Running Locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs

## 6. Running With Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

Open: http://localhost:8000/docs

## 7. Running Tests

```bash
python -m pytest
```

## 7a. API Demo

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4,
    "feature4": 0.2
  }'
```

**Example Response:**

```json
{
  "prediction": 0,
  "probability": 0.98
}
```

Swagger UI for interactive testing:

```
http://localhost:8000/docs
```

## 8. Dependency Management

All dependencies are **pinned to exact versions** in `requirements.txt` for reproducibility:

```
fastapi==0.129.0
uvicorn==0.40.0
scikit-learn==1.8.0
numpy==2.4.2
pandas==3.0.0
pytest==9.0.2
```

This ensures:
- ✅ Same packages across dev/staging/production
- ✅ Docker images are reproducible
- ✅ No surprise breaking changes

## 9. Engineering Decisions

- **Training separated from inference** — Reduces API complexity; training runs separately
- **Model artifacts persisted** — joblib serialization ensures compatibility
- **Input validation with Pydantic** — Type checks prevent malformed requests
- **Automated testing with pytest** — Full coverage ensures correctness
- **Containerized for reproducibility** — Docker eliminates "works on my machine"
- **No training inside API layer** — API stays lightweight and responsive
- **Auto-training fallback** — Container self-heals if models don't exist

This mirrors real-world ML backend architecture.

## 10. Future Improvements

- CI/CD pipeline
- Model versioning
- Cloud deployment (AWS/GCP/Azure)
- Monitoring & logging enhancements
