# app/pipelines/01_create_silver_tables.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
import random

import duckdb

DB_PATH = os.getenv("DUCKDB_PATH", "data/garv.duckdb")

AIRPORTS = ["DOH", "DXB", "LHR", "CDG", "FRA"]
REASONS = ["GATE_CHANGE", "LATE_ARRIVAL", "CREW", "TECH", "SECURITY", "WEATHER"]

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def create_silver_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS checkin_events (
        ts TIMESTAMP,
        airport VARCHAR,
        avg_wait_min DOUBLE,
        pax_count BIGINT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS presecurity_events (
        ts TIMESTAMP,
        airport VARCHAR,
        avg_wait_min DOUBLE,
        lanes_open DOUBLE,
        queue_len DOUBLE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS boarding_events (
        ts TIMESTAMP,
        airport VARCHAR,
        boarding_delay_min DOUBLE,
        reason_code VARCHAR
    );
    """)

def seed_dummy_data(conn: duckdb.DuckDBPyConnection, days: int = 14, rows_per_hour: int = 3) -> None:
    # Clear old demo data (safe for PoC)
    conn.execute("DELETE FROM checkin_events;")
    conn.execute("DELETE FROM presecurity_events;")
    conn.execute("DELETE FROM boarding_events;")

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)

    checkin_rows = []
    sec_rows = []
    boarding_rows = []

    ts = start
    while ts <= now:
        for ap in AIRPORTS:
            # multiple rows per hour to simulate raw events
            for _ in range(rows_per_hour):
                checkin_wait = max(0, random.gauss(12, 4))
                pax = max(10, int(random.gauss(220, 60)))

                sec_wait = max(0, random.gauss(18, 6))
                lanes = max(1, random.gauss(6, 1.5))
                qlen = max(0, random.gauss(120, 50))

                delay = max(0, random.gauss(7, 6))
                reason = random.choice(REASONS)

                checkin_rows.append((ts, ap, float(checkin_wait), int(pax)))
                sec_rows.append((ts, ap, float(sec_wait), float(lanes), float(qlen)))
                boarding_rows.append((ts, ap, float(delay), reason))

        ts += timedelta(hours=1)

    conn.executemany("INSERT INTO checkin_events VALUES (?, ?, ?, ?);", checkin_rows)
    conn.executemany("INSERT INTO presecurity_events VALUES (?, ?, ?, ?, ?);", sec_rows)
    conn.executemany("INSERT INTO boarding_events VALUES (?, ?, ?, ?);", boarding_rows)

def main():
    _ensure_dir(DB_PATH)
    conn = duckdb.connect(DB_PATH)

    create_silver_tables(conn)
    seed_dummy_data(conn, days=14, rows_per_hour=3)

    conn.close()
    print("✅ Silver tables created + dummy data loaded into:", DB_PATH)

if __name__ == "__main__":
    main()