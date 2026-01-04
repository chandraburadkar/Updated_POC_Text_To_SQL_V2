# app/pipelines/02_build_gold_tables.py
from __future__ import annotations

import os
import duckdb

DB_PATH = os.getenv("DUCKDB_PATH", "data/garv.duckdb")

def build_gold(conn: duckdb.DuckDBPyConnection) -> None:
    # 1) Hourly KPI
    conn.execute("DROP TABLE IF EXISTS gold_airport_kpi_hourly;")
    conn.execute("""
    CREATE TABLE gold_airport_kpi_hourly AS
    WITH
    c AS (
        SELECT
            airport,
            date_trunc('hour', ts) AS hour,
            AVG(avg_wait_min) AS checkin_wait_min,
            SUM(pax_count) AS pax_volume
        FROM checkin_events
        GROUP BY 1,2
    ),
    s AS (
        SELECT
            airport,
            date_trunc('hour', ts) AS hour,
            AVG(avg_wait_min) AS security_wait_min,
            AVG(lanes_open) AS avg_lanes_open,
            AVG(queue_len) AS avg_queue_len
        FROM presecurity_events
        GROUP BY 1,2
    ),
    b AS (
        SELECT
            airport,
            date_trunc('hour', ts) AS hour,
            AVG(boarding_delay_min) AS boarding_delay_min
        FROM boarding_events
        GROUP BY 1,2
    )
    SELECT
        c.airport,
        c.hour,
        c.checkin_wait_min,
        c.pax_volume,
        s.security_wait_min,
        s.avg_lanes_open,
        s.avg_queue_len,
        b.boarding_delay_min
    FROM c
    LEFT JOIN s USING (airport, hour)
    LEFT JOIN b USING (airport, hour);
    """)

    # 2) Daily top delay reason (by count)
    conn.execute("DROP TABLE IF EXISTS gold_delay_reason_daily;")
    conn.execute("""
    CREATE TABLE gold_delay_reason_daily AS
    WITH delay_daily AS (
        SELECT
            airport,
            CAST(ts AS DATE) AS day,
            reason_code,
            AVG(boarding_delay_min) AS avg_delay_min,
            COUNT(*) AS cnt
        FROM boarding_events
        GROUP BY 1,2,3
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY airport, day ORDER BY cnt DESC) AS rn
        FROM delay_daily
    )
    SELECT
        airport,
        day,
        reason_code AS top_reason,
        avg_delay_min AS top_reason_avg_delay_min,
        cnt AS top_reason_count
    FROM ranked
    WHERE rn = 1;
    """)

    # 3) Anomaly scores (z-score per airport on security_wait_min)
    conn.execute("DROP TABLE IF EXISTS gold_anomaly_scores;")
    conn.execute("""
    CREATE TABLE gold_anomaly_scores AS
    WITH k AS (
        SELECT *
        FROM gold_airport_kpi_hourly
        WHERE security_wait_min IS NOT NULL
    ),
    stats AS (
        SELECT
            airport,
            AVG(security_wait_min) AS mu,
            STDDEV_SAMP(security_wait_min) AS sigma
        FROM k
        GROUP BY 1
    )
    SELECT
        k.airport,
        k.hour AS ts,
        'security_wait_min' AS metric,
        ABS(
            (k.security_wait_min - s.mu)
            /
            CASE WHEN s.sigma IS NULL OR s.sigma = 0 THEN 1 ELSE s.sigma END
        ) AS score,
        (ABS(
            (k.security_wait_min - s.mu)
            /
            CASE WHEN s.sigma IS NULL OR s.sigma = 0 THEN 1 ELSE s.sigma END
        ) >= 2.5) AS is_anomaly,
        'demo_zscore_v1' AS model_version
    FROM k
    JOIN stats s USING (airport);
    """)

def main():
    conn = duckdb.connect(DB_PATH)
    build_gold(conn)
    conn.close()
    print("✅ Gold tables built in:", DB_PATH)

if __name__ == "__main__":
    main()