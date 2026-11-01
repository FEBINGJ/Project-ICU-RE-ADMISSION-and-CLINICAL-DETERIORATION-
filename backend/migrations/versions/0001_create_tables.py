"""create tables

Revision ID: 0001
Revises: nothing (the first migration)

Creates the 11 tables of Phase 7 Part B and the three id sequences.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ids for patients, stays and hospital admissions added through the app start at 90,000,001, above every MIMIC id.
    for name in ('patient_id_seq', 'stay_id_seq', 'hadm_id_seq'):
        op.execute(f'CREATE SEQUENCE {name} START WITH 90000001')

    op.create_table('labs',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('hadm_id', sa.BigInteger(), nullable=False),
    sa.Column('variable', sa.String(length=20), nullable=False),
    sa.Column('charttime', sa.DateTime(), nullable=False),
    sa.Column('avail_time', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.CheckConstraint("variable IN ('sodium', 'potassium', 'urea_nitrogen', 'creatinine', 'bicarbonate', 'hematocrit', 'glucose', 'wbc', 'platelets', 'hemoglobin', 'phosphate', 'calcium', 'lactate', 'inr')", name=op.f('ck_labs_variable')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_labs')),
    sa.UniqueConstraint('hadm_id', 'variable', 'charttime', name='uq_labs_hadm_variable_charttime')
    )
    op.create_table('patients',
    sa.Column('patient_id', sa.BigInteger(), autoincrement=False, nullable=False),
    sa.Column('age', sa.SmallInteger(), nullable=False),
    sa.Column('sex', sa.String(length=1), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("sex IN ('F', 'M')", name=op.f('ck_patients_sex')),
    sa.CheckConstraint('age >= 0 AND age <= 130', name=op.f('ck_patients_age_range')),
    sa.PrimaryKeyConstraint('patient_id', name=op.f('pk_patients'))
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('display_name', sa.String(length=100), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("role IN ('clinician', 'admin')", name=op.f('ck_users_role')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('username', name=op.f('uq_users_username'))
    )
    op.create_table('audit_log',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('username', sa.String(length=64), nullable=False),
    sa.Column('action', sa.String(length=40), nullable=False),
    sa.Column('object', sa.String(length=80), nullable=True),
    sa.Column('outcome', sa.String(length=10), nullable=False),
    sa.CheckConstraint("outcome IN ('ok', 'failed', 'denied')", name=op.f('ck_audit_log_outcome')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_audit_log_user_id_users'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_log'))
    )
    op.create_index('ix_audit_log_action', 'audit_log', ['action'], unique=False)
    op.create_index('ix_audit_log_at', 'audit_log', ['at'], unique=False)
    op.create_index('ix_audit_log_user_id', 'audit_log', ['user_id'], unique=False)
    op.create_table('replay_state',
    sa.Column('id', sa.SmallInteger(), nullable=False),
    sa.Column('replay_step', sa.Integer(), server_default='0', nullable=False),
    sa.Column('step_hours', sa.SmallInteger(), server_default='6', nullable=False),
    sa.Column('last_advanced_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_advanced_by_user_id', sa.Integer(), nullable=True),
    sa.CheckConstraint('id = 1', name=op.f('ck_replay_state_single_row')),
    sa.CheckConstraint('replay_step >= 0', name=op.f('ck_replay_state_step_not_negative')),
    sa.CheckConstraint('step_hours > 0', name=op.f('ck_replay_state_step_hours_positive')),
    sa.ForeignKeyConstraint(['last_advanced_by_user_id'], ['users.id'], name=op.f('fk_replay_state_last_advanced_by_user_id_users'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_replay_state'))
    )
    op.create_table('stays',
    sa.Column('stay_id', sa.BigInteger(), autoincrement=False, nullable=False),
    sa.Column('patient_id', sa.BigInteger(), nullable=False),
    sa.Column('hadm_id', sa.BigInteger(), nullable=False),
    sa.Column('stay_number', sa.SmallInteger(), nullable=False),
    sa.Column('care_unit', sa.String(length=80), nullable=False),
    sa.Column('last_care_unit', sa.String(length=80), nullable=True),
    sa.Column('intime', sa.DateTime(), nullable=False),
    sa.Column('outtime', sa.DateTime(), nullable=True),
    sa.Column('hosp_admit_time', sa.DateTime(), nullable=False),
    sa.Column('source', sa.String(length=20), nullable=False),
    sa.Column('died_in_icu', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('replay_start_time', sa.DateTime(), nullable=True),
    sa.Column('replay_time', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source <> 'dataset_replay' OR (outtime IS NOT NULL AND replay_start_time IS NOT NULL AND replay_time IS NOT NULL)", name=op.f('ck_stays_dataset_needs_replay')),
    sa.CheckConstraint("source IN ('dataset_replay', 'live_entry')", name=op.f('ck_stays_source')),
    sa.CheckConstraint('outtime IS NULL OR outtime >= intime', name=op.f('ck_stays_out_after_in')),
    sa.CheckConstraint('stay_number >= 1', name=op.f('ck_stays_stay_number_positive')),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], name=op.f('fk_stays_patient_id_patients')),
    sa.PrimaryKeyConstraint('stay_id', name=op.f('pk_stays')),
    sa.UniqueConstraint('patient_id', 'stay_number', name='uq_stays_patient_id_stay_number')
    )
    op.create_index(op.f('ix_stays_hadm_id'), 'stays', ['hadm_id'], unique=False)
    op.create_index(op.f('ix_stays_patient_id'), 'stays', ['patient_id'], unique=False)
    op.create_table('notes',
    sa.Column('note_id', sa.BigInteger(), nullable=False),
    sa.Column('stay_id', sa.BigInteger(), nullable=False),
    sa.Column('author_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('stay_hour', sa.Integer(), nullable=True),
    sa.Column('text', sa.Text(), nullable=False),
    sa.CheckConstraint('length(text) >= 1 AND length(text) <= 2000', name=op.f('ck_notes_text_length')),
    sa.ForeignKeyConstraint(['author_id'], ['users.id'], name=op.f('fk_notes_author_id_users')),
    sa.ForeignKeyConstraint(['stay_id'], ['stays.stay_id'], name=op.f('fk_notes_stay_id_stays'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('note_id', name=op.f('pk_notes'))
    )
    op.create_index('ix_notes_stay_id_created_at', 'notes', ['stay_id', 'created_at'], unique=False)
    op.create_table('predictions',
    sa.Column('prediction_id', sa.BigInteger(), nullable=False),
    sa.Column('stay_id', sa.BigInteger(), nullable=False),
    sa.Column('task', sa.String(length=20), nullable=False),
    sa.Column('made_at', sa.DateTime(), nullable=False),
    sa.Column('hours_since_icu_admit', sa.Float(), nullable=False),
    sa.Column('raw_score', sa.Float(), nullable=False),
    sa.Column('risk', sa.Float(), nullable=False),
    sa.Column('band', sa.String(length=10), nullable=True),
    sa.Column('above_alert_threshold', sa.Boolean(), nullable=False),
    sa.Column('news2', sa.SmallInteger(), nullable=True),
    sa.Column('mews', sa.SmallInteger(), nullable=True),
    sa.Column('model_source', sa.String(length=20), nullable=False),
    sa.Column('model_version', sa.String(length=100), nullable=False),
    sa.Column('explanation', postgresql.JSONB(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("band IS NULL OR band IN ('Low', 'Moderate', 'High')", name=op.f('ck_predictions_band')),
    sa.CheckConstraint("model_source IN ('cross_validated', 'live_model')", name=op.f('ck_predictions_model_source')),
    sa.CheckConstraint("task = 'deterioration' OR band IS NULL", name=op.f('ck_predictions_band_only_deterioration')),
    sa.CheckConstraint("task IN ('deterioration', 'readmission')", name=op.f('ck_predictions_task')),
    sa.CheckConstraint('risk >= 0 AND risk <= 1', name=op.f('ck_predictions_risk_range')),
    sa.ForeignKeyConstraint(['stay_id'], ['stays.stay_id'], name=op.f('fk_predictions_stay_id_stays'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('prediction_id', name=op.f('pk_predictions')),
    sa.UniqueConstraint('stay_id', 'task', 'made_at', name='uq_predictions_stay_task_made_at')
    )
    op.create_table('ventilation_periods',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('stay_id', sa.BigInteger(), nullable=False),
    sa.Column('start_time', sa.DateTime(), nullable=False),
    sa.Column('end_time', sa.DateTime(), nullable=True),
    sa.CheckConstraint('end_time IS NULL OR end_time > start_time', name=op.f('ck_ventilation_periods_end_after_start')),
    sa.ForeignKeyConstraint(['stay_id'], ['stays.stay_id'], name=op.f('fk_ventilation_periods_stay_id_stays'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ventilation_periods'))
    )
    op.create_index(op.f('ix_ventilation_periods_stay_id'), 'ventilation_periods', ['stay_id'], unique=False)
    op.create_table('vitals',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('stay_id', sa.BigInteger(), nullable=False),
    sa.Column('variable', sa.String(length=20), nullable=False),
    sa.Column('charttime', sa.DateTime(), nullable=False),
    sa.Column('avail_time', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.CheckConstraint("variable IN ('hr', 'rr', 'spo2', 'sbp', 'map', 'temp', 'gcs', 'o2_device')", name=op.f('ck_vitals_variable')),
    sa.ForeignKeyConstraint(['stay_id'], ['stays.stay_id'], name=op.f('fk_vitals_stay_id_stays'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_vitals')),
    sa.UniqueConstraint('stay_id', 'variable', 'charttime', name='uq_vitals_stay_variable_charttime')
    )
    op.create_table('alerts',
    sa.Column('alert_id', sa.BigInteger(), nullable=False),
    sa.Column('stay_id', sa.BigInteger(), nullable=False),
    sa.Column('prediction_id', sa.BigInteger(), nullable=True),
    sa.Column('alert_type', sa.String(length=30), nullable=False),
    sa.Column('raised_at', sa.DateTime(), nullable=False),
    sa.Column('hours_since_icu_admit', sa.Float(), nullable=True),
    sa.Column('risk', sa.Float(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='open', nullable=False),
    sa.Column('acknowledged_by_user_id', sa.Integer(), nullable=True),
    sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('rule_version', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("alert_type IN ('deterioration_high', 'readmission_flag')", name=op.f('ck_alerts_alert_type')),
    sa.CheckConstraint("status = 'open' OR (acknowledged_by_user_id IS NOT NULL AND acknowledged_at IS NOT NULL)", name=op.f('ck_alerts_acknowledged_has_who_and_when')),
    sa.CheckConstraint("status IN ('open', 'acknowledged')", name=op.f('ck_alerts_status')),
    sa.ForeignKeyConstraint(['acknowledged_by_user_id'], ['users.id'], name=op.f('fk_alerts_acknowledged_by_user_id_users'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['prediction_id'], ['predictions.prediction_id'], name=op.f('fk_alerts_prediction_id_predictions'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['stay_id'], ['stays.stay_id'], name=op.f('fk_alerts_stay_id_stays'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('alert_id', name=op.f('pk_alerts')),
    sa.UniqueConstraint('prediction_id', 'alert_type', name='uq_alerts_prediction_id_alert_type')
    )
    op.create_index('ix_alerts_status', 'alerts', ['status'], unique=False)
    op.create_index('ix_alerts_stay_id_raised_at', 'alerts', ['stay_id', 'raised_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_alerts_stay_id_raised_at', table_name='alerts')
    op.drop_index('ix_alerts_status', table_name='alerts')
    op.drop_table('alerts')
    op.drop_table('vitals')
    op.drop_index(op.f('ix_ventilation_periods_stay_id'), table_name='ventilation_periods')
    op.drop_table('ventilation_periods')
    op.drop_table('predictions')
    op.drop_index('ix_notes_stay_id_created_at', table_name='notes')
    op.drop_table('notes')
    op.drop_index(op.f('ix_stays_patient_id'), table_name='stays')
    op.drop_index(op.f('ix_stays_hadm_id'), table_name='stays')
    op.drop_table('stays')
    op.drop_table('replay_state')
    op.drop_index('ix_audit_log_user_id', table_name='audit_log')
    op.drop_index('ix_audit_log_at', table_name='audit_log')
    op.drop_index('ix_audit_log_action', table_name='audit_log')
    op.drop_table('audit_log')
    op.drop_table('users')
    op.drop_table('patients')
    op.drop_table('labs')
    for name in ('hadm_id_seq', 'stay_id_seq', 'patient_id_seq'):
        op.execute(f'DROP SEQUENCE {name}')
