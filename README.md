# AI-Based Clinical Risk Prediction System

Predictive analytics for **clinical deterioration** and **ICU readmission** using longitudinal EHR data, with SHAP explanations and an early-warning dashboard.

> Research and clinical decision-support prototype. It does not replace healthcare professionals, diagnose patients, or make treatment decisions.

## Pipeline

EHR data → Preprocessing → Health trends → Two prediction models → Risk assessment → Explanation → Early-warning dashboard

## Tech stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, NumPy |
| ML / DL | scikit-learn, XGBoost, PyTorch (LSTM) |
| Explainability | SHAP |
| Backend | FastAPI, PostgreSQL, SQLAlchemy |
| Frontend | React + TypeScript |
| Visualization | Matplotlib, Seaborn |

## Project structure

```
data/           raw, interim, processed (git-ignored, never committed)
notebooks/      EDA and experiments
ml/             data loading, features, models, explain, evaluation
backend/app/    FastAPI app (api, db, schemas, services)
frontend/       React + TypeScript app (created in Phase 8)
models/         saved trained models (git-ignored)
reports/        figures and result tables
docs/           scope note, literature notes, decisions
tests/          pytest tests
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your environment file
cp .env.example .env             # Windows: copy .env.example .env

# 4. Run the backend
uvicorn backend.app.main:app --reload

# 5. Run the tests
pytest
```

Then open http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs.

## Data policy

Patient data stays in `data/` and is **never pushed to GitHub**. Many clinical datasets (for example MIMIC) have licences that forbid redistribution. Only code and documentation are committed.

## Git workflow

- `main`: stable, working code only
- `dev`: integration branch
- `feature/<name>`: one branch per task (for example `feature/eda`, `feature/xgboost-baseline`)
- Commit messages: short and imperative, for example `Add rolling-window trend features`

## Roadmap

- [x] Phase 0: Setup
- [x] Phase 1: Literature review and research gap
- [x] Phase 2: Data understanding (EDA)
- [x] Phase 3: Cohort and label definition
- [x] Phase 4: Preprocessing and trend features
- [x] Phase 5: Modeling
- [x] Phase 6: Explainability
- [ ] Phase 7: Backend
- [ ] Phase 8: Frontend
- [ ] Phase 9: Integration and alerts
- [ ] Phase 10: Testing, documentation, report