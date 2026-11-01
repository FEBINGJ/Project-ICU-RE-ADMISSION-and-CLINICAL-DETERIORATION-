"""The 11 database tables (Phase 7 Part B). See docs/database.md for the diagram and the reasons behind each choice.

    users               who can log in (password stored only as a hash) and their role
    patients            one row per patient
    stays               one row per ICU stay; holds the replay clock for dataset stays
    vitals              vital-sign measurements (and oxygen-device records) per stay
    labs                lab results per hospital admission
    ventilation_periods invasive-ventilation episodes per stay
    predictions         every stored risk prediction with its explanation
    alerts              early-warning alerts and who acknowledged them
    notes               clinician notes per stay
    audit_log           who did what and when
    replay_state        one row: the global replay step

Two kinds of time are used on purpose:
    patient time (DateTime, no zone)      MIMIC dates are shifted decades into the past; they only order events.
    real time    (DateTime, with zone)    when a person actually did something (a note, a login).
Measurements keep both `charttime` (when taken) and `avail_time` (when a clinician could first see it). The replay
rule and the model features use `avail_time`, so nothing from the future leaks into what the dashboard shows.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer,
                        Sequence, SmallInteger, String, Text, UniqueConstraint, false, func, true)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

# ---- fixed vocabularies (the API contract, section 6). A test checks they match the ml code. ----
ROLES = ("clinician", "admin")
SEXES = ("F", "M")
STAY_SOURCES = ("dataset_replay", "live_entry")
TASKS = ("deterioration", "readmission")
BANDS = ("Low", "Moderate", "High")
MODEL_SOURCES = ("cross_validated", "live_model")
ALERT_TYPES = ("deterioration_high", "readmission_flag")
ALERT_STATUSES = ("open", "acknowledged")
AUDIT_OUTCOMES = ("ok", "failed", "denied")
VITAL_VARIABLES = ("hr", "rr", "spo2", "sbp", "map", "temp", "gcs", "o2_device")   # o2_device: any oxygen device charted
LAB_VARIABLES = ("sodium", "potassium", "urea_nitrogen", "creatinine", "bicarbonate", "hematocrit", "glucose", "wbc",
                 "platelets", "hemoglobin", "phosphate", "calcium", "lactate", "inr")

# Patients and stays added through the app get ids from 90,000,001 up, so they can never collide with MIMIC ids.
NEW_ID_START = 90_000_001
patient_id_seq = Sequence("patient_id_seq", start=NEW_ID_START, metadata=Base.metadata)
stay_id_seq = Sequence("stay_id_seq", start=NEW_ID_START, metadata=Base.metadata)
hadm_id_seq = Sequence("hadm_id_seq", start=NEW_ID_START, metadata=Base.metadata)   # hospital admission ids of live stays

BIG_ID = BigInteger().with_variant(Integer(), "sqlite")     # SQLite only auto-numbers INTEGER keys (used by the tests)
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


def _in(column: str, values: tuple) -> str:
    """SQL text 'column IN (...)' for a CHECK constraint."""
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=true())
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="role"),)


class Patient(Base):
    __tablename__ = "patients"
    patient_id: Mapped[int] = mapped_column(BIG_ID, patient_id_seq, primary_key=True)
    age: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sex: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    stays: Mapped[list["Stay"]] = relationship(back_populates="patient", order_by="Stay.stay_number")
    __table_args__ = (CheckConstraint(_in("sex", SEXES), name="sex"),
                      CheckConstraint("age >= 0 AND age <= 130", name="age_range"))


class Stay(Base):
    __tablename__ = "stays"
    stay_id: Mapped[int] = mapped_column(BIG_ID, stay_id_seq, primary_key=True)
    patient_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("patients.patient_id"), nullable=False, index=True)
    hadm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)       # hospital admission; labs hang on this
    stay_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)             # 1, 2, ... per patient, by time
    care_unit: Mapped[str] = mapped_column(String(80), nullable=False)                 # first ICU care unit
    last_care_unit: Mapped[Optional[str]] = mapped_column(String(80))                  # care unit at ICU discharge
    intime: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)              # ICU admission (patient time)
    outtime: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)                   # ICU discharge; empty while a live stay is open
    hosp_admit_time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)     # hospital admission (a readmission feature)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    died_in_icu: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())   # no readmission result then
    replay_start_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)         # where Reset puts the stay (dataset stays)
    replay_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)               # the stay's replay clock (dataset stays)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    patient: Mapped["Patient"] = relationship(back_populates="stays")
    __table_args__ = (
        UniqueConstraint("patient_id", "stay_number", name="uq_stays_patient_id_stay_number"),
        CheckConstraint(_in("source", STAY_SOURCES), name="source"),
        CheckConstraint("stay_number >= 1", name="stay_number_positive"),
        CheckConstraint("outtime IS NULL OR outtime >= intime", name="out_after_in"),
        CheckConstraint("source <> 'dataset_replay' OR (outtime IS NOT NULL AND replay_start_time IS NOT NULL "
                        "AND replay_time IS NOT NULL)", name="dataset_needs_replay"),
    )


class Vital(Base):
    __tablename__ = "vitals"
    id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    stay_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("stays.stay_id", ondelete="CASCADE"), nullable=False)
    variable: Mapped[str] = mapped_column(String(20), nullable=False)
    charttime: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)           # when it was measured
    avail_time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)          # when it could first be seen
    value: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (UniqueConstraint("stay_id", "variable", "charttime", name="uq_vitals_stay_variable_charttime"),
                      CheckConstraint(_in("variable", VITAL_VARIABLES), name="variable"))


class Lab(Base):
    __tablename__ = "labs"
    id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    hadm_id: Mapped[int] = mapped_column(BigInteger, nullable=False)                   # labs belong to the hospital admission
    variable: Mapped[str] = mapped_column(String(20), nullable=False)
    charttime: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    avail_time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (UniqueConstraint("hadm_id", "variable", "charttime", name="uq_labs_hadm_variable_charttime"),
                      CheckConstraint(_in("variable", LAB_VARIABLES), name="variable"))


class VentilationPeriod(Base):
    __tablename__ = "ventilation_periods"
    id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    stay_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("stays.stay_id", ondelete="CASCADE"), nullable=False, index=True)
    start_time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)                  # empty = still ventilated (live stays)
    __table_args__ = (CheckConstraint("end_time IS NULL OR end_time > start_time", name="end_after_start"),)


class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    stay_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("stays.stay_id", ondelete="CASCADE"), nullable=False)
    task: Mapped[str] = mapped_column(String(20), nullable=False)
    made_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)             # the time point (readmission: ICU discharge)
    hours_since_icu_admit: Mapped[float] = mapped_column(Float, nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)                    # the model's log-odds score
    risk: Mapped[float] = mapped_column(Float, nullable=False)                         # calibrated chance, 0 to 1
    band: Mapped[Optional[str]] = mapped_column(String(10))                            # deterioration only
    above_alert_threshold: Mapped[bool] = mapped_column(Boolean, nullable=False)
    news2: Mapped[Optional[int]] = mapped_column(SmallInteger)
    mews: Mapped[Optional[int]] = mapped_column(SmallInteger)
    model_source: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    explanation: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)               # factors, groups, narrative, warnings
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("stay_id", "task", "made_at", name="uq_predictions_stay_task_made_at"),
        CheckConstraint(_in("task", TASKS), name="task"),
        CheckConstraint("band IS NULL OR " + _in("band", BANDS), name="band"),
        CheckConstraint("task = 'deterioration' OR band IS NULL", name="band_only_deterioration"),
        CheckConstraint(_in("model_source", MODEL_SOURCES), name="model_source"),
        CheckConstraint("risk >= 0 AND risk <= 1", name="risk_range"),
    )


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    stay_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("stays.stay_id", ondelete="CASCADE"), nullable=False)
    prediction_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("predictions.prediction_id", ondelete="SET NULL"))
    alert_type: Mapped[str] = mapped_column(String(30), nullable=False)
    raised_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)           # patient time; shown only once replay reaches it
    hours_since_icu_admit: Mapped[Optional[float]] = mapped_column(Float)
    risk: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="open")
    acknowledged_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    acknowledged_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    rule_version: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("prediction_id", "alert_type", name="uq_alerts_prediction_id_alert_type"),
        CheckConstraint(_in("alert_type", ALERT_TYPES), name="alert_type"),
        CheckConstraint(_in("status", ALERT_STATUSES), name="status"),
        CheckConstraint("status = 'open' OR (acknowledged_by_user_id IS NOT NULL AND acknowledged_at IS NOT NULL)",
                        name="acknowledged_has_who_and_when"),
        Index("ix_alerts_stay_id_raised_at", "stay_id", "raised_at"),
        Index("ix_alerts_status", "status"),
    )


class Note(Base):
    __tablename__ = "notes"
    note_id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    stay_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("stays.stay_id", ondelete="CASCADE"), nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    stay_hour: Mapped[Optional[int]] = mapped_column(Integer)                          # hour of the stay when it was written
    text: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (CheckConstraint("length(text) >= 1 AND length(text) <= 2000", name="text_length"),
                      Index("ix_notes_stay_id_created_at", "stay_id", "created_at"))


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BIG_ID, primary_key=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    username: Mapped[str] = mapped_column(String(64), nullable=False)                  # kept even for failed logins of unknown users
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    object: Mapped[Optional[str]] = mapped_column(String(80))
    outcome: Mapped[str] = mapped_column(String(10), nullable=False)
    __table_args__ = (CheckConstraint(_in("outcome", AUDIT_OUTCOMES), name="outcome"),
                      Index("ix_audit_log_at", "at"), Index("ix_audit_log_user_id", "user_id"),
                      Index("ix_audit_log_action", "action"))


class ReplayState(Base):
    __tablename__ = "replay_state"
    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)                    # always 1: there is one global clock
    replay_step: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    step_hours: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="6")
    last_advanced_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    last_advanced_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),
                      CheckConstraint("replay_step >= 0", name="step_not_negative"),
                      CheckConstraint("step_hours > 0", name="step_hours_positive"))
