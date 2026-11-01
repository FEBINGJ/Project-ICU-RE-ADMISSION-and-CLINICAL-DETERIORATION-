"""Tests for the database layer (Phase 7 Part B).

They run on a temporary in-memory SQLite database, so they need no PostgreSQL and never touch your real data.
(The migration itself is checked against real PostgreSQL with `alembic upgrade head` and `alembic check`.)
"""
import datetime as dt
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, event, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.config import Settings
from backend.app.db import models as m
from backend.app.db.base import Base

T0 = dt.datetime(2154, 4, 24, 23, 3, 44)          # shifted MIMIC-style time
HOUR = dt.timedelta(hours=1)
EXPECTED_TABLES = {"users", "patients", "stays", "vitals", "labs", "ventilation_periods", "predictions", "alerts",
                   "notes", "audit_log", "replay_state"}


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))   # SQLite ignores FKs unless asked
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def make_user(db, name="demo", role="clinician"):
    u = m.User(username=name, password_hash="not-a-real-hash", display_name=name.title(), role=role)
    db.add(u)
    db.flush()
    return u


def make_patient(db, pid=10018328, age=83, sex="F"):
    p = m.Patient(patient_id=pid, age=age, sex=sex)
    db.add(p)
    db.flush()
    return p


def make_stay(db, stay_id=31269608, pid=10018328, number=1, hadm=23786647):
    if db.get(m.Patient, pid) is None:
        make_patient(db, pid)
    s = m.Stay(stay_id=stay_id, patient_id=pid, hadm_id=hadm, stay_number=number, care_unit="Neuro Stepdown",
               intime=T0, outtime=T0 + 40 * HOUR, hosp_admit_time=T0 - 2 * HOUR, source="dataset_replay",
               replay_start_time=T0 + 18 * HOUR, replay_time=T0 + 18 * HOUR)
    db.add(s)
    db.flush()
    return s


def make_prediction(db, stay_id=31269608, task="deterioration", hours=18, band="Moderate", risk=0.061):
    p = m.Prediction(stay_id=stay_id, task=task, made_at=T0 + hours * HOUR, hours_since_icu_admit=hours, raw_score=0.5,
                     risk=risk, band=band if task == "deterioration" else None, above_alert_threshold=False,
                     model_source="cross_validated", model_version="test", explanation={"factors": []})
    db.add(p)
    db.flush()
    return p


# ---------------------------------------------------------------- structure
def test_the_eleven_tables_exist(db):
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_columns_the_api_contract_needs():
    """docs/api_contract.md sections 4 and 8: what the replay rule and the pages read."""
    need = {"vitals": {"stay_id", "variable", "charttime", "avail_time", "value"},
            "labs": {"hadm_id", "variable", "charttime", "avail_time", "value"},
            "stays": {"intime", "outtime", "hosp_admit_time", "last_care_unit", "hadm_id", "source", "died_in_icu",
                      "replay_time", "replay_start_time"},
            "predictions": {"made_at", "hours_since_icu_admit", "raw_score", "risk", "band", "above_alert_threshold",
                            "news2", "mews", "model_source", "model_version", "explanation"},
            "alerts": {"alert_type", "raised_at", "risk", "message", "status", "acknowledged_by_user_id",
                       "acknowledged_at", "rule_version"},
            "notes": {"author_id", "created_at", "stay_hour", "text"},
            "audit_log": {"at", "user_id", "username", "action", "object", "outcome"},
            "replay_state": {"replay_step", "step_hours", "last_advanced_at", "last_advanced_by_user_id"}}
    for table, columns in need.items():
        assert columns <= set(Base.metadata.tables[table].columns.keys()), table


def test_vocabularies_match_the_ml_code():
    from ml.features.clean import LABS_CORE, LABS_EXTRA
    from ml.features.windows import VITAL_VARS
    assert set(m.VITAL_VARIABLES) - {"o2_device"} == set(VITAL_VARS)
    assert set(m.LAB_VARIABLES) == set(LABS_CORE) | set(LABS_EXTRA)


def test_new_ids_start_above_every_mimic_id():
    assert m.NEW_ID_START == 90_000_001
    for seq in (m.patient_id_seq, m.stay_id_seq, m.hadm_id_seq):
        assert seq.start == m.NEW_ID_START


def test_migration_file_matches_the_models():
    """The hand-checked migration must create exactly the tables and sequences the models define."""
    path = Path(__file__).resolve().parents[1] / "backend" / "migrations" / "versions" / "0001_create_tables.py"
    source = path.read_text()
    assert set(re.findall(r"op\.create_table\('(\w+)'", source)) == EXPECTED_TABLES
    assert set(re.findall(r"op\.drop_table\('(\w+)'", source)) == EXPECTED_TABLES
    assert all(name in source for name in ("patient_id_seq", "stay_id_seq", "hadm_id_seq"))


# ---------------------------------------------------------------- rules the database enforces
BAD_ROWS = {
    "user with an unknown role": lambda db: m.User(username="x", password_hash="h", display_name="X", role="nurse"),
    "patient with sex X": lambda db: m.Patient(patient_id=1, age=40, sex="X"),
    "patient aged 200": lambda db: m.Patient(patient_id=1, age=200, sex="F"),
    "stay with an unknown source": lambda db: m.Stay(patient_id=make_patient(db).patient_id, hadm_id=1, stay_number=1,
                                                     care_unit="MICU", intime=T0, hosp_admit_time=T0, source="imported"),
    "stay that ends before it starts": lambda db: m.Stay(patient_id=make_patient(db).patient_id, hadm_id=1, stay_number=1,
                                                         care_unit="MICU", intime=T0, outtime=T0 - HOUR,
                                                         hosp_admit_time=T0, source="live_entry"),
    "dataset stay without replay times": lambda db: m.Stay(patient_id=make_patient(db).patient_id, hadm_id=1, stay_number=1,
                                                           care_unit="MICU", intime=T0, outtime=T0 + HOUR,
                                                           hosp_admit_time=T0, source="dataset_replay"),
    "vital with an unknown variable": lambda db: m.Vital(stay_id=make_stay(db).stay_id, variable="pulse",
                                                         charttime=T0, avail_time=T0, value=80),
    "lab with an unknown variable": lambda db: m.Lab(hadm_id=1, variable="magic", charttime=T0, avail_time=T0, value=1),
    "ventilation that ends before it starts": lambda db: m.VentilationPeriod(stay_id=make_stay(db).stay_id,
                                                                             start_time=T0 + HOUR, end_time=T0),
    "readmission prediction with a band": lambda db: m.Prediction(
        stay_id=make_stay(db).stay_id, task="readmission", made_at=T0 + 40 * HOUR, hours_since_icu_admit=40, raw_score=0,
        risk=0.2, band="High", above_alert_threshold=True, model_source="live_model", model_version="t", explanation={}),
    "prediction with risk above 1": lambda db: m.Prediction(
        stay_id=make_stay(db).stay_id, task="deterioration", made_at=T0 + 6 * HOUR, hours_since_icu_admit=6, raw_score=0,
        risk=1.5, above_alert_threshold=False, model_source="live_model", model_version="t", explanation={}),
    "prediction with an unknown band": lambda db: m.Prediction(
        stay_id=make_stay(db).stay_id, task="deterioration", made_at=T0 + 6 * HOUR, hours_since_icu_admit=6, raw_score=0,
        risk=0.1, band="Extreme", above_alert_threshold=False, model_source="live_model", model_version="t", explanation={}),
    "acknowledged alert with nobody who acknowledged it": lambda db: m.Alert(
        stay_id=make_stay(db).stay_id, alert_type="deterioration_high", raised_at=T0, risk=0.08, message="m",
        status="acknowledged", rule_version="v1"),
    "alert with an unknown type": lambda db: m.Alert(stay_id=make_stay(db).stay_id, alert_type="loud", raised_at=T0,
                                                     risk=0.08, message="m", rule_version="v1"),
    "empty note": lambda db: m.Note(stay_id=make_stay(db).stay_id, author_id=make_user(db).id, text=""),
    "note longer than 2000 characters": lambda db: m.Note(stay_id=make_stay(db).stay_id, author_id=make_user(db).id,
                                                          text="x" * 2001),
    "audit row with an unknown outcome": lambda db: m.AuditLog(username="x", action="login", outcome="maybe"),
    "a second replay_state row": lambda db: m.ReplayState(id=2),
}


@pytest.mark.parametrize("label", sorted(BAD_ROWS))
def test_database_rejects(db, label):
    row = BAD_ROWS[label](db)
    with pytest.raises(IntegrityError, match="CHECK constraint failed"):        # rejected by the rule, not by a missing value
        db.add(row)
        db.flush()


def test_database_accepts_the_good_versions(db):
    stay = make_stay(db)
    make_prediction(db)
    make_prediction(db, task="readmission", hours=40, band=None, risk=0.18)
    db.add_all([m.Vital(stay_id=stay.stay_id, variable="o2_device", charttime=T0, avail_time=T0, value=1),
                m.Lab(hadm_id=stay.hadm_id, variable="platelets", charttime=T0, avail_time=T0 + HOUR, value=33),
                m.VentilationPeriod(stay_id=stay.stay_id, start_time=T0 + 5 * HOUR),          # still ventilated: no end yet
                m.ReplayState(id=1)])
    db.flush()


def test_duplicates_are_rejected(db):
    stay = make_stay(db)
    make_prediction(db)
    user = make_user(db)
    attempts = [lambda: m.User(username=user.username, password_hash="h", display_name="D", role="admin"),
                lambda: m.Stay(patient_id=stay.patient_id, hadm_id=2, stay_number=stay.stay_number, care_unit="x",
                               intime=T0, hosp_admit_time=T0, source="live_entry"),
                lambda: make_prediction_row(stay.stay_id)]
    db.add_all([m.Vital(stay_id=stay.stay_id, variable="hr", charttime=T0, avail_time=T0, value=80),
                m.Lab(hadm_id=stay.hadm_id, variable="sodium", charttime=T0, avail_time=T0, value=139)])
    db.flush()
    attempts += [lambda: m.Vital(stay_id=stay.stay_id, variable="hr", charttime=T0, avail_time=T0, value=81),
                 lambda: m.Lab(hadm_id=stay.hadm_id, variable="sodium", charttime=T0, avail_time=T0, value=140)]
    for build in attempts:
        with pytest.raises(IntegrityError):
            with db.begin_nested():
                db.add(build())
                db.flush()


def make_prediction_row(stay_id):
    return m.Prediction(stay_id=stay_id, task="deterioration", made_at=T0 + 18 * HOUR, hours_since_icu_admit=18, raw_score=0,
                        risk=0.1, above_alert_threshold=False, model_source="live_model", model_version="t", explanation={})


def test_foreign_keys_are_enforced(db):
    with pytest.raises(IntegrityError):
        db.add(m.Stay(patient_id=999, hadm_id=1, stay_number=1, care_unit="x", intime=T0, hosp_admit_time=T0,
                      source="live_entry"))
        db.flush()


def test_deleting_a_stay_deletes_everything_attached_to_it(db):
    stay = make_stay(db)
    user = make_user(db)
    pred = make_prediction(db)
    db.add_all([m.Vital(stay_id=stay.stay_id, variable="hr", charttime=T0, avail_time=T0, value=80),
                m.VentilationPeriod(stay_id=stay.stay_id, start_time=T0),
                m.Note(stay_id=stay.stay_id, author_id=user.id, text="hello"),
                m.Alert(stay_id=stay.stay_id, prediction_id=pred.prediction_id, alert_type="deterioration_high",
                        raised_at=T0, risk=0.08, message="m", rule_version="v1")])
    db.flush()
    db.execute(delete(m.Stay).where(m.Stay.stay_id == stay.stay_id))
    for table in (m.Vital, m.VentilationPeriod, m.Note, m.Alert, m.Prediction):
        assert db.execute(select(table)).first() is None, table.__tablename__


# ---------------------------------------------------------------- defaults and data types
def test_defaults(db):
    user = make_user(db)
    stay = make_stay(db)
    alert = m.Alert(stay_id=stay.stay_id, alert_type="readmission_flag", raised_at=T0, risk=0.2, message="m", rule_version="v1")
    db.add(alert)
    db.flush()
    db.refresh(user)
    db.refresh(stay)
    db.refresh(alert)
    assert user.is_active is True                 # a new account can log in
    assert alert.status == "open"                 # a new alert is open
    assert stay.died_in_icu is False
    assert user.created_at is not None


def test_explanation_is_stored_as_json(db):
    stay = make_stay(db)
    pred = make_prediction(db)
    pred.explanation = {"factors": [{"label": "Latest platelet count", "value": 33.0}], "warnings": ["use as context"]}
    db.flush()
    db.expire_all()
    got = db.get(m.Prediction, pred.prediction_id).explanation
    assert got["factors"][0]["label"] == "Latest platelet count" and got["warnings"] == ["use as context"]
    assert stay.patient_id == 10018328


def test_patient_stays_are_ordered_by_stay_number(db):
    make_stay(db, stay_id=2, number=2, hadm=2)
    make_stay(db, stay_id=1, number=1, hadm=1)
    db.expire_all()
    assert [s.stay_number for s in db.get(m.Patient, 10018328).stays] == [1, 2]


# ---------------------------------------------------------------- settings
def _clean_env(monkeypatch):
    for name in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_password_with_special_characters_survives(monkeypatch):
    _clean_env(monkeypatch)
    password = "p@ss:w/rd#1 %"
    url = Settings(_env_file=None, db_password=password).sqlalchemy_url
    assert url.password == password
    assert make_url(url.render_as_string(hide_password=False)).password == password      # survives being written as text
    assert url.drivername == "postgresql+psycopg2" and url.database == "icu_risk" and url.port == 5432


def test_empty_password_gives_a_clear_message(monkeypatch):
    _clean_env(monkeypatch)
    with pytest.raises(RuntimeError, match=r"\.env"):
        Settings(_env_file=None).sqlalchemy_url


def test_database_url_replaces_the_separate_values(monkeypatch):
    _clean_env(monkeypatch)
    s = Settings(_env_file=None, database_url="postgresql+psycopg2://u:secret@db.example:5433/other")
    assert (s.sqlalchemy_url.host, s.sqlalchemy_url.port, s.sqlalchemy_url.database) == ("db.example", 5433, "other")


def test_describe_never_shows_the_password(monkeypatch):
    _clean_env(monkeypatch)
    assert "secret" not in Settings(_env_file=None, db_password="secret").describe()
    assert "secret" not in Settings(_env_file=None, database_url="postgresql://u:secret@h/d").describe()
