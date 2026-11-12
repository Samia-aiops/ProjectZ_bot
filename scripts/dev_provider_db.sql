-- Dev-only fixture for the provider Text-to-SQL analytics feature.
-- Rows for tenant 'default' / provider 'dev-provider-1' match the identity that
-- dev_principal("provider") produces (DEV_BYPASS_ROLE=provider). The decoy rows
-- exist so an isolation regression shows up as another provider's data leaking.
-- This file only runs when the provider-db volume is empty.

CREATE TABLE provider (
    id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    firstname VARCHAR(100) NOT NULL,
    lastname VARCHAR(100) NOT NULL,
    status VARCHAR(32) NOT NULL,
    verification_status VARCHAR(32) NOT NULL,
    degree VARCHAR(64),
    npi VARCHAR(32),
    practice_role VARCHAR(64),
    practice_name VARCHAR(200),
    practice_size VARCHAR(64),
    booking_type VARCHAR(64),
    primary_specialty VARCHAR(100),
    rating DECIMAL(3, 2),
    reviews_count INT NOT NULL DEFAULT 0,
    tenant_id VARCHAR(64) NOT NULL,
    PRIMARY KEY (id, tenant_id)
);

CREATE TABLE appointment_analytics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    period VARCHAR(32) NOT NULL,
    today_count INT NOT NULL DEFAULT 0,
    this_week_count INT NOT NULL DEFAULT 0,
    last_week_count INT NOT NULL DEFAULT 0,
    this_month_count INT NOT NULL DEFAULT 0,
    last_month_count INT NOT NULL DEFAULT 0,
    this_year_count INT NOT NULL DEFAULT 0,
    last_year_count INT NOT NULL DEFAULT 0,
    start_date DATE,
    end_date DATE,
    selected_count INT NOT NULL DEFAULT 0,
    KEY idx_provider_tenant (provider_id, tenant_id)
);

CREATE TABLE availability (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    working_days VARCHAR(100) NOT NULL,
    appointment_availability TINYINT(1) NOT NULL DEFAULT 1,
    weekly_working_schedule_json JSON,
    KEY idx_provider_tenant (provider_id, tenant_id)
);

CREATE TABLE appointments (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,
    patient_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    appointment_type VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    appointment_date DATE NOT NULL,
    created_at DATETIME NOT NULL,
    KEY idx_provider_tenant (provider_id, tenant_id)
);

CREATE TABLE insurances (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    insurance_name VARCHAR(100) NOT NULL,
    KEY idx_provider_tenant (provider_id, tenant_id)
);

INSERT INTO provider VALUES
    ('dev-provider-1', 'user-1', 'Ada', 'Lovelace', 'active', 'verified', 'MD', '1234567890',
     'owner', 'Analytical Practice', '2-5', 'instant', 'Internal Medicine', 4.80, 42, 'default'),
    ('other-provider-1', 'user-2', 'Grace', 'Hopper', 'active', 'verified', 'MD', '0987654321',
     'member', 'Compiler Clinic', '6-10', 'scheduled', 'Cardiology', 4.60, 91, 'default'),
    ('dev-provider-1', 'user-3', 'Alan', 'Turing', 'active', 'verified', 'MD', '1112223333',
     'owner', 'Bletchley Practice', '2-5', 'instant', 'Neurology', 4.10, 7, 'tenant-b');

INSERT INTO appointment_analytics
    (provider_id, tenant_id, period, today_count, this_week_count, last_week_count, this_month_count,
     last_month_count, this_year_count, last_year_count, start_date, end_date, selected_count)
VALUES
    ('dev-provider-1', 'default', 'today', 4, 11, 9, 27, 31, 240, 198, '2026-11-12', '2026-11-12', 4),
    ('dev-provider-1', 'default', 'month', 4, 11, 9, 27, 31, 240, 198, '2026-11-01', '2026-11-30', 27),
    ('other-provider-1', 'default', 'today', 9, 40, 38, 150, 140, 900, 870, '2026-11-12', '2026-11-12', 9),
    ('dev-provider-1', 'tenant-b', 'today', 2, 5, 6, 12, 15, 120, 99, '2026-11-12', '2026-11-12', 2);

INSERT INTO availability (provider_id, tenant_id, working_days, appointment_availability, weekly_working_schedule_json)
VALUES
    ('dev-provider-1', 'default', 'Mon,Tue,Wed,Thu,Fri', 1,
     '{"mon":["09:00","17:00"],"tue":["09:00","17:00"],"wed":["09:00","17:00"],"thu":["09:00","17:00"],"fri":["09:00","15:00"]}'),
    ('other-provider-1', 'default', 'Mon,Wed,Fri', 0, '{"mon":["10:00","16:00"]}');

INSERT INTO appointments (provider_id, patient_id, tenant_id, appointment_type, status, appointment_date, created_at)
VALUES
    ('dev-provider-1', 'patient-1', 'default', 'follow_up', 'completed', '2026-11-10', '2026-11-01 09:00:00'),
    ('dev-provider-1', 'patient-2', 'default', 'new_patient', 'completed', '2026-11-11', '2026-11-02 10:30:00'),
    ('dev-provider-1', 'patient-3', 'default', 'follow_up', 'scheduled', '2026-11-12', '2026-11-05 14:15:00'),
    ('dev-provider-1', 'patient-4', 'default', 'telehealth', 'cancelled', '2026-11-12', '2026-11-06 11:45:00'),
    ('other-provider-1', 'patient-9', 'default', 'new_patient', 'scheduled', '2026-11-12', '2026-11-07 08:00:00');

INSERT INTO insurances (provider_id, tenant_id, insurance_name)
VALUES
    ('dev-provider-1', 'default', 'Aetna'),
    ('dev-provider-1', 'default', 'Cigna'),
    ('other-provider-1', 'default', 'BlueCross');
