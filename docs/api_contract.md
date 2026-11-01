# API Contract (Phase 7, Part A)

Status: design only. No backend code exists for these endpoints yet. Parts B to F build them, and every later part is checked against this file.
Mockup: the five-artboard "ICU Risk Dashboard Mockup" (login, dashboard, patient overview, risk analysis, add patient). Every value shown there is sample data for layout.

This is a research prototype on the de-identified MIMIC-IV demo. Nothing here is for clinical use. All numbers in the examples below are samples.

---

## 1. Decisions this contract is built on

Please confirm or change these before Part B, because they decide the database tables.

| # | Decision | Why |
|---|---|---|
| D1 | The dashboard list has **one row per ICU stay** (140 rows), showing the patient ID and the stay number. The patient page has a stay switcher. | Risk, replay clock and alerts all belong to a stay. A patient with two stays would otherwise need rules for "which stay is current". |
| D2 | **Replay per stay.** Each stay has its own `replay_time`. One "Advance" moves every stay forward by 6 hours of patient time (the model's 6-hour step). A clock stops at the stay's discharge. | The models predict on a 6-hour grid. Keeping the step equal to the grid means every step shows exactly one new prediction. |
| D3 | **Auto-play is a browser timer.** The server has no background job: it only has `POST /monitoring/advance` and `POST /monitoring/reset`. "Pause" only stops the browser timer. | Keeps the backend simple and testable. |
| D4 | **Starting position:** each stay starts half-way through (rounded down to a 6-hour step), so the first dashboard already shows stays in the ICU and stays that ended. Reset returns to this. | A dashboard where every patient is at hour 6 and nobody is discharged would not show the readmission side. |
| D5 | **Visibility rule (no future leaks).** The API shows a record only when it was already available at the stay's replay time. See section 4. | Otherwise the dashboard would show the future. |
| D6 | **Readmission is predicted once, at ICU discharge.** Before that the API says `pending`. It returns a percentage and a flag (`Flagged` / `Not flagged`), and no Low/Moderate/High band. A stay that ended in death in the ICU (12 of the 140) gets `not_applicable`: that patient cannot be readmitted. | The model is trained at discharge using the whole stay. Phase 6 found the readmission bands did not separate cases reliably. |
| D7 | **Deterioration bands** are Low / Moderate / High, always sent with the observed rate from cross-validation. NEWS2 is sent next to them. | Phase 6: the bands are supported but the gap is small (3.3 / 3.8 / 5.3%). NEWS2 bands separated better, so show both honestly. |
| D8 | Every stored prediction has a `model_source`: `cross_validated` (dataset stays) or `live_model` (new patients). | The saved models were fitted on all 100 patients, so using them on dataset patients would show overconfident in-sample risks. |
| D9 | **Alerts:** `deterioration_high` is raised when a stay's risk enters the High band from a lower band (a rising edge). `readmission_flag` is raised at discharge when the readmission risk is Flagged. Cool-down and wording are fixed in Phase 9. Each alert stores its `rule_version`. | 45% of all time points fall in High, so alerting on every High row would flood the screen (alert fatigue). |
| D10 | **Dates are shifted** (MIMIC dates are moved decades back). Every time is sent as an ISO string plus `hours_since_icu_admit`. The screens lead with the hours. | Avoids reading the shifted date as the real date. |
| D11 | **Roles:** `clinician` (read, notes, acknowledge, add patient, add measurement, discharge) and `admin` (all of that, plus advance/reset replay and the audit log). | Matches the mockup. |
| D12 | **Add patient is one API call** at the end of the three-step form. The steps live in the browser; nothing is saved before step 3. | No half-created patients. |
| D13 | **New patients are "live" stays** (`source: "live_entry"`): no replay clock, their "now" is the newest measurement. Measurements and discharge can be added only to live stays. Dataset stays return `409`. | The dataset stays are fixed history. |

---

## 2. Conventions

- Base path: `/api/v1`. The existing `GET /health` stays outside it.
- JSON in and out, UTF-8. Interactive docs are served by FastAPI at `/docs`.
- Auth: `Authorization: Bearer <token>` on everything except `POST /auth/login` and `GET /health`.
- Times: ISO 8601 without a zone for patient time (shifted, e.g. `2154-04-25T17:03:44`). Times of real actions (notes, acknowledgements, audit log) are UTC with `Z`.
- Risks: `risk` is a fraction (0.061) and `risk_percent` the same as a percent with one decimal (6.1). Both are always sent.
- Missing is `null`, never `0`, `""` or `"N/A"`.
- IDs are integers. Dataset patients and stays keep their MIMIC IDs. New patients and stays get IDs from 90000001 up.
- Lists use `page` (from 1), `page_size` (default 20, maximum 100). Response: `{"items": [...], "page": 1, "page_size": 20, "total": 140}`.
- Sorting: `sort=deterioration_risk` or `sort=-deterioration_risk` (minus = descending). Allowed fields are listed per endpoint.
- Errors always look like this, with the HTTP status code set to match:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Heart rate must be between 20 and 250.",
    "details": [{"field": "vitals.hr", "problem": "out_of_range", "allowed": [20, 250]}]
  }
}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Malformed request |
| 401 | `not_authenticated` | Missing, expired or wrong token, or wrong username/password |
| 403 | `forbidden` | Role not allowed (for example a clinician calls `/monitoring/advance`) |
| 404 | `not_found` | Unknown ID, **or a stay/record that is not visible yet at the replay time** |
| 409 | `conflict` | Dataset stay used for a live-only action, discharge twice, alert already acknowledged |
| 422 | `validation_error` | Field-level problems (as above) |

---

## 3. Screen to data map

| Screen | Element | Endpoint | Fields used |
|---|---|---|---|
| Login | Form, errors | `POST /auth/login` | `access_token`, `user.role`, `user.display_name` |
| App bar | Name and role, Alerts badge | `GET /auth/me`, `GET /dashboard/summary` | `display_name`, `role`, `open_alerts` |
| Dashboard | Replay bar | `GET /monitoring/state` | `replay_step`, `step_hours`; buttons call `POST /monitoring/advance`, `/reset` (admin) |
| Dashboard | Five total tiles | `GET /dashboard/summary` | `stays_total`, `patients_total`, `in_icu`, `discharged`, `deterioration_high`, `readmission_flagged`, `open_alerts` |
| Dashboard | Patient table, search, filters, sort, paging | `GET /stays` | every field of the stay row (section 7.4) |
| Dashboard | Alerts feed, Acknowledge | `GET /alerts?status=open&page_size=4`, `PATCH /alerts/{id}` | alert fields |
| Patient header | Demographics, status, stay switcher | `GET /stays/{id}`, `GET /patients/{id}` | `stay`, `other_stays` |
| Overview | Deterioration card, risk mini-trend | `GET /stays/{id}`, `GET /stays/{id}/predictions?task=deterioration` | `deterioration`, history list |
| Overview | Readmission card (pending or result) | `GET /stays/{id}` | `readmission` |
| Overview | NEWS2 / MEWS | `GET /stays/{id}` | `scores` |
| Overview | 48-hour trend tiles | `GET /stays/{id}/vitals?hours=48` | `series` |
| Overview | Top 5 factors, Alerts for this stay | `GET /stays/{id}/risk`, `GET /alerts?stay_id=` | `factors`, alert list |
| Vital signs tab | Full charts and table | `GET /stays/{id}/vitals?hours=48` (or `hours=all`) | `series[].points` |
| Lab results tab | Table with time and value | `GET /stays/{id}/labs?hours=all` | `series[].points` |
| Risk analysis tab | Summary, bar, band note, warnings | `GET /stays/{id}/risk?task=` | `risk`, `average_risk_percent`, `band`, `band_note`, `warnings`, `model_source` |
| Risk analysis tab | Factor bars, groups, words, reading guide | same | `factors`, `groups`, `rest`, `narrative`, `reading_guide` |
| Risk analysis tab | Prediction history, About the model | `GET /stays/{id}/predictions`, same `model_info` | `history`, `model_info` |
| Notes tab | List, add note | `GET /stays/{id}/notes`, `POST /stays/{id}/notes` | note fields |
| Add patient | Save (step 3) | `POST /patients` | request and response in 7.13 |
| Add patient | New measurements later, discharge | `POST /stays/{id}/vitals`, `/labs`, `/discharge` | 7.13 |
| Admin | Audit log | `GET /admin/audit-log` | audit fields |

---

## 4. The replay (visibility) rule

For each stay the server keeps one `replay_time` (patient time). Every endpoint filters with it:

| Record | Visible when |
|---|---|
| Vital sign, lab result | `avail_time <= replay_time`. Use `avail_time`, when the value could first be seen, never `charttime`. This is the same rule the features use. |
| Deterioration prediction | its time point `t <= replay_time` |
| Readmission prediction | `replay_time >= outtime` (the stay has ended) |
| Alert | `raised_at <= replay_time` |
| Stay `outtime`, `discharge_location`, ICU length of stay | only after the stay has ended |
| Stay status | `not_started` if `replay_time < intime`, `in_icu` if `replay_time < outtime`, else `discharged` |

Never sent by any endpoint: the training labels (`y`, `readmit_30d`, `readmit_7d`, `readmit_48h`, `vent_next`, `death_next`), `died_after_icu_in_hospital`, and anything dated after `replay_time`. A test in Part E checks that, for random stays and times, no response field is later than `replay_time`.

Live stays (D13) have no replay clock. Their "now" is the newest measurement, and the same rules apply with that time.

---

## 5. Endpoint list

| Method | Path | Role | Purpose | Built in |
|---|---|---|---|---|
| GET | `/health` | none | Server alive | exists |
| POST | `/auth/login` | none | Get a token | D |
| GET | `/auth/me` | any | Current user | D |
| GET | `/dashboard/summary` | any | Totals and counts | E |
| GET | `/monitoring/state` | any | Replay step | E |
| POST | `/monitoring/advance` | admin | Move every stay forward 6 h (`steps` 1 to 4) | F |
| POST | `/monitoring/reset` | admin | Back to the starting positions | F |
| GET | `/stays` | any | Patient list (search, filter, sort, paging) | E |
| GET | `/stays/{stay_id}` | any | Header, current predictions, scores | E |
| GET | `/stays/{stay_id}/vitals` | any | Vital-sign series | E |
| GET | `/stays/{stay_id}/labs` | any | Lab series | E |
| GET | `/stays/{stay_id}/risk` | any | Full explanation of one prediction | E |
| GET | `/stays/{stay_id}/predictions` | any | Prediction history (no explanations) | E |
| GET | `/patients/{patient_id}` | any | Demographics and all stays | E |
| GET | `/alerts` | any | Alert list (filters) | F |
| PATCH | `/alerts/{alert_id}` | clinician, admin | Acknowledge | F |
| GET | `/stays/{stay_id}/notes` | any | Notes | F |
| POST | `/stays/{stay_id}/notes` | clinician, admin | Add note | F |
| POST | `/patients` | clinician, admin | Add new patient with first measurements | F |
| POST | `/stays/{stay_id}/vitals` | clinician, admin | Add vital signs (live stays) | F |
| POST | `/stays/{stay_id}/labs` | clinician, admin | Add lab results (live stays) | F |
| POST | `/stays/{stay_id}/discharge` | clinician, admin | End a live stay and calculate readmission risk | F |
| GET | `/admin/audit-log` | admin | Who did what | D |

Every call to `/stays/{id}...`, `/patients/...`, notes, alerts and measurement endpoints writes an audit-log row (user, action, object, time).

---

## 6. Vocabulary (fixed values)

| Name | Values |
|---|---|
| `status` (stay) | `not_started`, `in_icu`, `discharged` |
| `task` | `deterioration`, `readmission` |
| `band` (deterioration only) | `Low`, `Moderate`, `High` |
| `flag` (readmission only) | `Flagged`, `Not flagged` |
| `readmission.status` | `pending` (stay not ended yet), `available`, `not_applicable` (the patient died in the ICU) |
| `model_source` | `cross_validated`, `live_model` |
| `source` (stay) | `dataset_replay`, `live_entry` |
| `alert_type` | `deterioration_high`, `readmission_flag` |
| `alert status` | `open`, `acknowledged` |
| `role` | `clinician`, `admin` |
| `sex` | `F`, `M` |
| `direction` (factor) | `raises`, `lowers` |
| `strength` (factor) | `large`, `medium`, `small` |
| `news2.band` | `Low` (0 to 4), `Medium` (5 to 6), `High` (7 or more) |
| factor `group` | `Latest vital signs`, `Vital-sign trends`, `Latest lab results`, `Lab trends (24 h)`, `Measurement pattern`, `Whole-stay vitals`, `Patient and stay context` |
| `care_unit` | `Cardiac Vascular Intensive Care Unit (CVICU)`, `Coronary Care Unit (CCU)`, `Medical Intensive Care Unit (MICU)`, `Medical/Surgical Intensive Care Unit (MICU/SICU)`, `Neuro Intermediate`, `Neuro Stepdown`, `Neuro Surgical Intensive Care Unit (Neuro SICU)`, `Surgical Intensive Care Unit (SICU)`, `Trauma SICU (TSICU)` |

Accepted ranges for entered measurements (values outside return `422`; they match the cleaning rules the model was trained with):

| Vitals | Range | Labs | Range |
|---|---|---|---|
| `hr` bpm | 20 to 250 | `sodium` mmol/L | 100 to 180 |
| `rr` breaths/min | 4 to 60 | `potassium` mmol/L | 1.5 to 9 |
| `spo2` % | 50 to 100 | `urea_nitrogen` mg/dL | 1 to 200 |
| `sbp` mmHg | 40 to 300 | `creatinine` mg/dL | 0.1 to 20 |
| `map` mmHg | 20 to 200 | `bicarbonate` mmol/L | 2 to 50 |
| `temp` °C | 30 to 43.5 | `hematocrit` % | 5 to 70 |
| `gcs` points | 3 to 15 | `glucose` mg/dL | 20 to 1500 |
| `on_vent`, `supplemental_o2` | true / false | `wbc` K/µL | 0.1 to 200 |
| | | `platelets` K/µL | 1 to 1500 |
| | | `hemoglobin` g/dL | 2 to 25 |
| | | `phosphate` mg/dL | 0.1 to 20 |
| | | `calcium` mg/dL | 3 to 20 |
| | | `lactate` mmol/L (optional) | 0.2 to 30 |
| | | `inr` (optional) | 0.5 to 15 |

---

## 7. Request and response shapes

### 7.1 `POST /auth/login`

Request:

```json
{"username": "clinician.demo", "password": "from-the-env-file"}
```

Response `200`:

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {"id": 1, "username": "clinician.demo", "display_name": "Demo Clinician", "role": "clinician"}
}
```

Wrong username or password: `401 not_authenticated` with the same message for both cases. `expires_in` (seconds) is fixed in Part D.

### 7.2 `GET /auth/me`

```json
{"id": 1, "username": "clinician.demo", "display_name": "Demo Clinician", "role": "clinician"}
```

### 7.3 `GET /dashboard/summary` and `GET /monitoring/state`

```json
{
  "replay_step": 14,
  "stays_total": 140,
  "patients_total": 100,
  "in_icu": 63,
  "discharged": 77,
  "not_started": 0,
  "deterioration_high": 28,
  "readmission_flagged": 35,
  "open_alerts": 7
}
```

`deterioration_high` counts stays currently in the ICU whose latest band is High. `readmission_flagged` counts discharged stays whose readmission result is Flagged.

```json
{
  "replay_step": 14,
  "step_hours": 6,
  "last_advanced_at": "2026-11-01T10:12:09Z",
  "last_advanced_by": "admin.demo"
}
```

`POST /monitoring/advance` takes `{"steps": 1}` and returns the same object plus `"new_alerts": 2`. `POST /monitoring/reset` returns the same object with `replay_step` 0.

### 7.4 `GET /stays`

Query: `q` (patient ID, starts with), `status`, `band` (`Low|Moderate|High`, deterioration), `care_unit`, `alerts_open=true`, `sort` (`deterioration_risk`, `open_alerts`, `hours_in_icu`, `patient_id`; default `-deterioration_risk`), `page`, `page_size`.

```json
{
  "items": [
    {
      "stay_id": 31269608,
      "patient_id": 10018328,
      "stay_number": 1,
      "n_stays": 2,
      "age": 83,
      "sex": "F",
      "care_unit": "Neuro Stepdown",
      "status": "in_icu",
      "source": "dataset_replay",
      "hours_in_icu": 18,
      "deterioration": {
        "risk": 0.061,
        "risk_percent": 6.1,
        "band": "Moderate",
        "above_alert_threshold": false,
        "delta_percent_points": 0.8,
        "made_at": "2154-04-25T17:03:44",
        "hours_since_icu_admit": 18
      },
      "readmission": {"status": "pending", "risk_percent": null, "flag": null},
      "news2": {"value": 3, "band": "Low"},
      "open_alerts": 0
    },
    {
      "stay_id": 37067082,
      "patient_id": 10001217,
      "stay_number": 1,
      "n_stays": 1,
      "age": 55,
      "sex": "F",
      "care_unit": "Surgical Intensive Care Unit (SICU)",
      "status": "discharged",
      "source": "dataset_replay",
      "hours_in_icu": 27,
      "deterioration": null,
      "readmission": {"status": "available", "risk_percent": 21.0, "flag": "Flagged"},
      "news2": {"value": 4, "band": "Low"},
      "open_alerts": 1
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 140
}
```

Notes: `deterioration` is `null` for a discharged stay (the stay ended) and for a stay with no prediction yet (the first prediction is at hour 6). `news2` is the value at the latest time point, or at discharge for a discharged stay. `delta_percent_points` is the change since the previous step, `null` at the first prediction.

### 7.5 `GET /stays/{stay_id}`

```json
{
  "stay": {
    "stay_id": 31269608,
    "patient_id": 10018328,
    "stay_number": 1,
    "n_stays": 2,
    "age": 83,
    "sex": "F",
    "care_unit": "Neuro Stepdown",
    "status": "in_icu",
    "source": "dataset_replay",
    "intime": "2154-04-24T23:03:44",
    "outtime": null,
    "hours_in_icu": 18,
    "replay_time": "2154-04-25T17:03:44",
    "replay_step": 14
  },
  "other_stays": [
    {"stay_id": 33448811, "stay_number": 2, "status": "not_started"}
  ],
  "deterioration": {
    "prediction_id": 8812,
    "task": "deterioration",
    "made_at": "2154-04-25T17:03:44",
    "hours_since_icu_admit": 18,
    "risk": 0.061,
    "risk_percent": 6.1,
    "average_risk_percent": 4.4,
    "band": "Moderate",
    "band_note": "In cross-validation, 3.8% of cases in the Moderate band had the event (95% range 2.2% to 5.4%); the average for everyone is 4.4%.",
    "above_alert_threshold": false,
    "flag": null,
    "delta_percent_points": 0.8,
    "model_source": "cross_validated"
  },
  "readmission": {
    "status": "pending",
    "message": "Calculated at ICU discharge."
  },
  "scores": {
    "news2": {"value": 3, "band": "Low", "computed_at": "2154-04-25T17:03:44"},
    "mews": {"value": 2, "computed_at": "2154-04-25T17:03:44"}
  }
}
```

When the readmission result exists, `readmission` has the same fields as `deterioration` above with `task` `readmission`, `band` `null`, `flag` `"Flagged"` or `"Not flagged"`, and a `band_note` that says why bands are not shown. `outtime` is `null` until the stay has ended. For a `not_started` stay, `deterioration`, `readmission` and `scores` are `null`. A stay that ended in death in the ICU has `{"status": "not_applicable", "message": "The patient died in the ICU, so a readmission cannot happen."}` as its `readmission` here, and `{"status": "not_applicable", "risk_percent": null, "flag": null}` in the list of 7.4.

### 7.6 `GET /stays/{stay_id}/vitals?hours=48`

`hours` is a number (default 48, maximum 240) or `all`. The window ends at `replay_time`.

```json
{
  "stay_id": 31269608,
  "window_hours": 48,
  "as_of": "2154-04-25T17:03:44",
  "series": [
    {
      "variable": "hr",
      "label": "Heart rate",
      "unit": "bpm",
      "latest": {"time": "2154-04-25T16:30:00", "hours_since_icu_admit": 17.5, "value": 88},
      "min": 76,
      "max": 104,
      "points": [
        {"time": "2154-04-25T15:30:00", "hours_since_icu_admit": 16.5, "value": 91},
        {"time": "2154-04-25T16:30:00", "hours_since_icu_admit": 17.5, "value": 88}
      ]
    }
  ],
  "flags": {"on_vent": false, "supplemental_o2": false}
}
```

`series` contains `hr`, `rr`, `spo2`, `sbp`, `map`, `temp`, `gcs`. A variable with no values yet has `points: []` and `latest: null`. `time` is `charttime`.

### 7.7 `GET /stays/{stay_id}/labs?hours=all`

Same shape as vitals, with `variable` from the lab names in section 6. Labs belong to the hospital admission, so results taken before ICU admission are included and have a negative `hours_since_icu_admit`.

```json
{
  "stay_id": 31269608,
  "window_hours": "all",
  "as_of": "2154-04-25T17:03:44",
  "series": [
    {
      "variable": "platelets",
      "label": "Platelet count",
      "unit": "K/µL",
      "latest": {"time": "2154-04-25T04:10:00", "hours_since_icu_admit": 5.1, "value": 33},
      "min": 33,
      "max": 74,
      "points": [
        {"time": "2154-04-24T19:20:00", "hours_since_icu_admit": -3.7, "value": 74},
        {"time": "2154-04-25T04:10:00", "hours_since_icu_admit": 5.1, "value": 33}
      ]
    }
  ]
}
```

### 7.8 `GET /stays/{stay_id}/risk?task=deterioration`

`task` is required. Optional `prediction_id` shows an earlier prediction. This is the output of the Phase 6 explanation code (`ml/explain/explain.py`) with the same field names, plus the prediction fields and `model_info` that the backend adds. One field is renamed: the code's integer `alert` (0 or 1) is sent as the boolean `above_alert_threshold`. Readmission is `404` while the stay is not discharged.

```json
{
  "prediction_id": 8812,
  "task": "deterioration",
  "made_at": "2154-04-25T17:03:44",
  "hours_since_icu_admit": 18,
  "risk": 0.061,
  "risk_percent": 6.1,
  "average_risk_percent": 4.4,
  "band": "Moderate",
  "band_note": "In cross-validation, 3.8% of cases in the Moderate band had the event (95% range 2.2% to 5.4%); the average for everyone is 4.4%.",
  "above_alert_threshold": false,
  "model_source": "cross_validated",
  "raw_score": 0.54,
  "base_score": 0.36,
  "factors": [
    {
      "feature": "rr__std_24h",
      "label": "Variability (standard deviation) of breathing rate over the last 24 h",
      "value": 5.05,
      "value_text": "5 breaths/min",
      "group": "Vital-sign trends",
      "note": "",
      "contribution": 0.306,
      "direction": "raises",
      "share": 0.073,
      "strength": "medium"
    },
    {
      "feature": "platelets__last",
      "label": "Latest platelet count",
      "value": 33.0,
      "value_text": "33 K/µL",
      "group": "Latest lab results",
      "note": "",
      "contribution": 0.234,
      "direction": "raises",
      "share": 0.056,
      "strength": "medium"
    }
  ],
  "groups": [
    {"group": "Vital-sign trends", "net_contribution": 1.39, "direction": "raises"},
    {"group": "Patient and stay context", "net_contribution": -0.41, "direction": "lowers"}
  ],
  "rest": {"n_inputs": 140, "net_contribution": -0.14},
  "narrative": "The model estimates a 6.1% chance of starting invasive ventilation or dying within the next 24 hours (the average in the training data is 4.4%). This falls in the Moderate band.\n\nFactors that pushed the score UP: ...",
  "reading_guide": "The model starts from an average score. Each factor below moves the score up (raises the risk) or down (lowers it); ...",
  "warnings": [
    "Model discrimination is close to chance (cross-validated AUROC 0.59, 95% range 0.47 to 0.71): use as context, not as a decision."
  ],
  "model_info": {
    "model": "XGBoost",
    "feature_set": "C",
    "trained_on": "MIMIC-IV Clinical Database Demo 2.2, 100 patients",
    "cv_auroc": 0.586,
    "cv_auroc_ci": [0.47, 0.71],
    "verdict": "No clear winner against NEWS2 and MEWS in cross-validation."
  }
}
```

The example shows two of the five factors to stay short; the real response has `top_k` = 5 (default). For readmission, `band` is `null` and `flag` is added. The numbers inside `groups`, `rest` and `raw_score` are sample values.

For `model_source: "cross_validated"` the explanation comes from the fold model that never saw this patient. For `live_model` it comes from the saved final model.

### 7.9 `GET /stays/{stay_id}/predictions?task=deterioration`

```json
{
  "stay_id": 31269608,
  "task": "deterioration",
  "items": [
    {"prediction_id": 8812, "made_at": "2154-04-25T17:03:44", "hours_since_icu_admit": 18, "risk_percent": 6.1, "band": "Moderate", "above_alert_threshold": false, "news2": 3, "model_source": "cross_validated"},
    {"prediction_id": 8811, "made_at": "2154-04-25T11:03:44", "hours_since_icu_admit": 12, "risk_percent": 5.3, "band": "Moderate", "above_alert_threshold": false, "news2": 3, "model_source": "cross_validated"}
  ]
}
```

Newest first. Only predictions visible at the replay time.

### 7.10 `GET /patients/{patient_id}`

```json
{
  "patient_id": 10018328,
  "age": 83,
  "sex": "F",
  "stays": [
    {"stay_id": 31269608, "stay_number": 1, "care_unit": "Neuro Stepdown", "status": "in_icu"},
    {"stay_id": 33448811, "stay_number": 2, "care_unit": "Neuro Stepdown", "status": "not_started"}
  ]
}
```

### 7.11 Alerts

`GET /alerts` takes `status`, `stay_id`, `alert_type`, `page`, `page_size` and returns newest first.

```json
{
  "items": [
    {
      "alert_id": 501,
      "stay_id": 31269608,
      "patient_id": 10018328,
      "alert_type": "deterioration_high",
      "raised_at": "2154-04-25T17:03:44",
      "hours_since_icu_admit": 18,
      "risk_percent": 8.0,
      "message": "Deterioration risk entered the High band (8.0%).",
      "status": "open",
      "acknowledged_by": null,
      "acknowledged_at": null,
      "rule_version": "v1"
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 7
}
```

`PATCH /alerts/{alert_id}` with `{"status": "acknowledged"}` returns the updated alert (`acknowledged_by` is the user's display name, `acknowledged_at` is real UTC time). A second acknowledgement is `409`.

### 7.12 Notes

`GET /stays/{stay_id}/notes`, newest first:

```json
{
  "items": [
    {
      "note_id": 77,
      "stay_id": 31269608,
      "author": {"id": 1, "display_name": "Demo Clinician"},
      "created_at": "2026-11-01T10:20:41Z",
      "stay_hour": 18,
      "text": "Breathing rate more variable overnight; reviewed with the team."
    }
  ]
}
```

`POST /stays/{stay_id}/notes` takes `{"text": "..."}` (1 to 2000 characters) and returns the created note with `201`. `stay_hour` is filled by the server from the replay time. Notes are not edited or deleted in this prototype.

### 7.13 `POST /patients` (add new patient)

Request. Every vital and lab is optional (omit or `null` = not measured). `basic` is required.

```json
{
  "basic": {
    "age": 61,
    "sex": "F",
    "care_unit": "Medical Intensive Care Unit (MICU)",
    "admission_time": "2154-05-02T08:00:00"
  },
  "vitals": {
    "hr": 88, "rr": 18, "spo2": 96, "sbp": 118, "map": 78, "temp": 36.8, "gcs": 15,
    "on_vent": false, "supplemental_o2": false
  },
  "labs": {
    "sodium": 139, "potassium": 4.1, "urea_nitrogen": 18, "creatinine": 0.9, "bicarbonate": 24,
    "hematocrit": 36, "glucose": 110, "wbc": 9.5, "platelets": 230, "hemoglobin": 12.5,
    "phosphate": 3.4, "calcium": 8.9, "lactate": null, "inr": null
  }
}
```

Response `201`:

```json
{
  "patient": {"patient_id": 90000001, "age": 61, "sex": "F"},
  "stay": {"stay_id": 90000101, "stay_number": 1, "status": "in_icu", "source": "live_entry", "care_unit": "Medical Intensive Care Unit (MICU)", "intime": "2154-05-02T08:00:00"},
  "deterioration": {
    "prediction_id": 9001,
    "task": "deterioration",
    "made_at": "2154-05-02T08:00:00",
    "hours_since_icu_admit": 0,
    "risk": 0.052,
    "risk_percent": 5.2,
    "average_risk_percent": 4.4,
    "band": "Moderate",
    "band_note": "In cross-validation, 3.8% of cases in the Moderate band had the event (95% range 2.2% to 5.4%); the average for everyone is 4.4%.",
    "above_alert_threshold": false,
    "flag": null,
    "delta_percent_points": null,
    "model_source": "live_model"
  },
  "warnings": [
    "This prediction was made at hour 0. The model was trained on time points from hour 6 onward, so treat it with extra care."
  ],
  "alert": null
}
```

`alert` holds an alert object (as in 7.11) when the new prediction is in the High band. Validation errors list every bad field in `details`.

### 7.14 Measurements and discharge (live stays only)

`POST /stays/{stay_id}/vitals`: `{"time": "2154-05-02T14:00:00", "vitals": {"hr": 96, "rr": 22, "on_vent": false}}`. `POST /stays/{stay_id}/labs`: `{"time": "...", "labs": {"platelets": 190}}`. Both return `201` with the same body shape as 7.13 minus `patient` and `stay` (the new `deterioration` prediction, `warnings`, `alert`). `time` must be after the stay's `intime` and not before the previous measurement.

`POST /stays/{stay_id}/discharge`: `{"time": "2154-05-04T09:00:00"}`. Returns `201` with the stay (now `discharged`) and the `readmission` result in the shape of 7.5. A second discharge is `409`.

On a dataset stay all three return `409 conflict`.

### 7.15 `GET /admin/audit-log`

Query: `user`, `action`, `from`, `to` (UTC), `page`, `page_size`.

```json
{
  "items": [
    {"id": 9001, "at": "2026-11-01T10:20:41Z", "user": "clinician.demo", "action": "note.create", "object": "stay:31269608", "outcome": "ok"},
    {"id": 9000, "at": "2026-11-01T10:12:09Z", "user": "admin.demo", "action": "replay.advance", "object": "steps:1", "outcome": "ok"},
    {"id": 8999, "at": "2026-11-01T10:05:12Z", "user": "unknown", "action": "login", "object": "clinician.demo", "outcome": "failed"}
  ],
  "page": 1,
  "page_size": 20,
  "total": 3
}
```

Actions logged: `login`, `patient.view`, `note.create`, `alert.acknowledge`, `patient.create`, `measurement.create`, `stay.discharge`, `replay.advance`, `replay.reset`. Passwords and tokens are never logged.

---

## 8. What Part B must be able to store

Derived from the contract so the tables do not need changing later.

| Table | Key contents |
|---|---|
| `users` | username, password hash, display name, role |
| `patients` | patient ID, age, sex |
| `stays` | stay ID, patient ID, hospital admission ID, stay number, first and last care unit, `intime`, `outtime`, hospital admission time, `source`, `died_in_icu`, `replay_start_time`, `replay_time` (the last two for dataset stays only) |
| `vitals` | stay ID, variable (the 7 vital signs plus `o2_device`, any oxygen device charted), `charttime`, `avail_time`, value |
| `labs` | hospital admission ID (a stay finds its labs through its admission), variable, `charttime`, `avail_time`, value |
| `ventilation_periods` | stay ID, start time, end time (empty while still ventilated). The model's "on a ventilator" inputs are worked out from these and from `o2_device` |
| `predictions` | stay ID, task, `made_at`, hours since admission, `raw_score`, `risk`, `band`, `above_alert_threshold`, `news2`, `mews`, `model_source`, `model_version`, explanation as JSON |
| `alerts` | stay ID, type, `raised_at`, risk, message, status, acknowledged by/at, `rule_version` |
| `notes` | stay ID, author, `created_at`, `stay_hour`, text |
| `audit_log` | time, user, action, object, outcome |
| `replay_state` | one row: step, step hours, last advanced at/by |

Size check: about 2,100 stored predictions (1,986 deterioration time points plus 128 readmission results) each with an explanation of a few KB, so roughly 10 to 20 MB. PostgreSQL handles this without trouble.

---

## 9. Known limits (shown honestly in the product)

- Both models are close to chance in cross-validation (AUROC about 0.59). They did not clearly beat NEWS2 or MEWS. The screens say so.
- Deterioration bands are different but only slightly in observed rate (3.3%, 3.8%, 5.3%). Readmission bands are not shown.
- Cross-validated predictions are only available for the 100 dataset patients. Predictions for new patients come from the final model, which is not validated on them.
- A new patient entered at admission is predicted at hour 0, earlier than any training row (hour 6 onward). The API warns about it.
- The dataset has 100 patients, so none of this proves clinical usefulness.