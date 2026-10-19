from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Clinical Risk Prediction API",
    description="Research prototype: deterioration and ICU readmission risk from EHR data.",
    version="0.1.0",
)

# Allow the React dev server (Vite default port) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Simple check that the API is running."""
    return {"status": "ok"}
