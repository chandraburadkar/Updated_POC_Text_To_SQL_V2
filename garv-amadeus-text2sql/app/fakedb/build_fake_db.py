# app/fakedb/build_fake_db.py
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.db.duckdb_client import get_conn

AIRPORTS = [
    ("LHR", "London Heathrow"),
    ("CDG", "Paris Charles de Gaulle"),
    ("FRA", "Frankfurt"),
    ("AMS", "Amsterdam Schiphol"),
    ("DXB", "Dubai"),
]

REASON_CODES = [
    ("CREW", "Crew late"),
    ("GATE", "Gate change / congestion"),
    ("WX", "Weather"),
    ("SEC", "Security disruption"),
    ("ATC", "Air traffic control"),
    ("BAG", "Baggage loading"),
    ("TECH", "Technical issue"),
]

def _rand_choice_weighted(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

def _ts_range(start_utc: datetime, hours: int, step_minutes: int = 5):
    cur = start_utc
    end = start_utc + timedelta(hours=hours)
    while cur <= end:
        yield cur
        cur += timedelta(minutes=step_minutes)

def create_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dim_airport (
        airport STRING PRIMARY KEY,
        airport_name STRING
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS flights (
        flight_id STRING,
        airport STRING,
        airline STRING,
        flight_number STRING,
        scheduled_departure TIMESTAMP,
        actual_departure TIMESTAMP,
        destination STRING,
        status STRING,
        PRIMARY KEY (flight_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS checkin_events (
        airport STRING,
        ts TIMESTAMP,
        pax_count INTEGER,
        avg_wait_min DOUBLE,
        counters_open INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS presecurity_events (
        airport STRING,
        ts TIMESTAMP,
        pax_count INTEGER,
        avg_wait_min DOUBLE,
        lanes_open INTEGER,
        queue_len INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS boarding_events (
        airport STRING,
        ts TIMESTAMP,
        flight_id STRING,
        boarding_delay_min DOUBLE,
        reason_code STRING
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS disruption_events (
        airport STRING,
        ts TIMESTAMP,
        disruption_type STRING,          -- e.g., SECURITY, WEATHER, ATC, SYSTEM
        severity STRING,                 -- LOW/MED/HIGH
        impacted_area STRING,            -- CHECKIN/SECURITY/BOARDING
        notes STRING
    );
    """)

def seed_dimensions(conn):
    df_airport = pd.DataFrame(AIRPORTS, columns=["airport", "airport_name"])
    conn.execute("DELETE FROM dim_airport;")
    conn.register("df_airport", df_airport)
    conn.execute("INSERT INTO dim_airport SELECT * FROM df_airport;")
    conn.unregister("df_airport")

def gen_flights(start_utc: datetime, days: int = 7):
    airlines = ["BA", "AF", "LH", "KL", "EK"]
    destinations = ["JFK", "SFO", "SIN", "DEL", "HKG", "MAD", "ROM"]

    rows = []
    for d in range(days):
        day = start_utc.date() + timedelta(days=d)
        for (ap, _) in AIRPORTS:
            for i in range(25):  # flights per airport per day
                airline = random.choice(airlines)
                fn = f"{airline}{random.randint(100, 9999)}"
                dep_hour = random.randint(5, 23)
                dep_min = random.choice([0, 10, 20, 30, 40, 50])
                sched = datetime(day.year, day.month, day.day, dep_hour, dep_min, tzinfo=timezone.utc)

                # actual departure (delay distribution)
                delay = max(0, random.gauss(mu=8, sigma=15))  # avg ~8 min
                if random.random() < 0.08:
                    delay += random.randint(20, 90)           # occasional big delays

                actual = sched + timedelta(minutes=delay)

                status = "DEPARTED" if actual <= datetime.now(timezone.utc) else "SCHEDULED"
                dest = random.choice(destinations)

                flight_id = f"{ap}-{fn}-{int(sched.timestamp())}"

                rows.append({
                    "flight_id": flight_id,
                    "airport": ap,
                    "airline": airline,
                    "flight_number": fn,
                    "scheduled_departure": sched.replace(tzinfo=None),
                    "actual_departure": actual.replace(tzinfo=None),
                    "destination": dest,
                    "status": status
                })
    return pd.DataFrame(rows)

def gen_ops_events(start_utc: datetime, hours: int = 24 * 7):
    """
    Generate time-series events every 5 minutes.
    Introduce rush-hour effects + occasional disruptions.
    """
    checkin_rows = []
    sec_rows = []
    boarding_rows = []
    disruption_rows = []

    # Define airport base volumes (rough)
    base_volume = {"LHR": 110, "CDG": 95, "FRA": 90, "AMS": 85, "DXB": 120}

    for ap, _ in AIRPORTS:
        for ts in _ts_range(start_utc, hours=hours, step_minutes=5):
            hour = ts.hour
            # rush hours: morning + evening
            rush_multiplier = 1.0
            if 6 <= hour <= 10:
                rush_multiplier = 1.6
            elif 16 <= hour <= 20:
                rush_multiplier = 1.5
            elif 0 <= hour <= 4:
                rush_multiplier = 0.6

            pax = int(max(0, random.gauss(base_volume[ap] * rush_multiplier, 18)))
            counters_open = int(max(6, min(35, random.gauss(14 * rush_multiplier, 3))))
            lanes_open = int(max(3, min(18, random.gauss(7 * rush_multiplier, 2))))
            queue_len = int(max(0, random.gauss(pax * 0.6, pax * 0.25)))

            # waits correlated with load vs capacity
            checkin_wait = max(0, (pax / max(1, counters_open)) * random.uniform(0.6, 1.2))
            security_wait = max(0, (queue_len / max(1, lanes_open)) * random.uniform(0.4, 0.9))

            # disruptions: low probability but increases wait
            if random.random() < 0.01:
                disruption_type = _rand_choice_weighted(
                    ["SECURITY", "WEATHER", "ATC", "SYSTEM"],
                    [0.35, 0.25, 0.25, 0.15]
                )
                severity = _rand_choice_weighted(["LOW", "MED", "HIGH"], [0.5, 0.35, 0.15])
                impacted_area = _rand_choice_weighted(["CHECKIN", "SECURITY", "BOARDING"], [0.2, 0.6, 0.2])

                disruption_rows.append({
                    "airport": ap,
                    "ts": ts.replace(tzinfo=None),
                    "disruption_type": disruption_type,
                    "severity": severity,
                    "impacted_area": impacted_area,
                    "notes": f"{disruption_type} event - {severity}"
                })

                # impact
                if impacted_area == "SECURITY":
                    security_wait *= random.uniform(1.6, 2.6)
                    queue_len = int(queue_len * random.uniform(1.3, 1.9))
                elif impacted_area == "CHECKIN":
                    checkin_wait *= random.uniform(1.4, 2.2)
                else:
                    security_wait *= random.uniform(1.1, 1.4)

            checkin_rows.append({
                "airport": ap,
                "ts": ts.replace(tzinfo=None),
                "pax_count": pax,
                "avg_wait_min": round(checkin_wait, 2),
                "counters_open": counters_open
            })

            sec_rows.append({
                "airport": ap,
                "ts": ts.replace(tzinfo=None),
                "pax_count": pax,
                "avg_wait_min": round(security_wait, 2),
                "lanes_open": lanes_open,
                "queue_len": queue_len
            })

    # boarding events: sample from flights later (join in SQL)
    # For now create generic boarding delay events at 15-min granularity
    for ap, _ in AIRPORTS:
        for ts in _ts_range(start_utc, hours=hours, step_minutes=15):
            # fewer boarding events
            if random.random() < 0.4:
                delay = max(0, random.gauss(4, 8))
                # sometimes heavy delay
                if random.random() < 0.06:
                    delay += random.randint(20, 80)

                reason = _rand_choice_weighted(
                    [rc[0] for rc in REASON_CODES],
                    [0.12, 0.18, 0.10, 0.12, 0.16, 0.18, 0.14]
                )

                boarding_rows.append({
                    "airport": ap,
                    "ts": ts.replace(tzinfo=None),
                    "flight_id": None,  # will be enriched later if needed
                    "boarding_delay_min": round(delay, 2),
                    "reason_code": reason
                })

    return (
        pd.DataFrame(checkin_rows),
        pd.DataFrame(sec_rows),
        pd.DataFrame(boarding_rows),
        pd.DataFrame(disruption_rows),
    )

def load_to_duckdb(conn, table: str, df: pd.DataFrame, overwrite: bool = True):
    if overwrite:
        conn.execute(f"DELETE FROM {table};")
    conn.register("df_temp", df)
    conn.execute(f"INSERT INTO {table} SELECT * FROM df_temp;")
    conn.unregister("df_temp")

def main():
    random.seed(42)
    conn = get_conn()

    print("✅ Connected to DuckDB")
    create_tables(conn)
    seed_dimensions(conn)

    # 7 days window ending now
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)

    flights_df = gen_flights(start_utc=start, days=7)
    checkin_df, sec_df, boarding_df, disruption_df = gen_ops_events(start_utc=start, hours=24 * 7)

    load_to_duckdb(conn, "flights", flights_df, overwrite=True)
    load_to_duckdb(conn, "checkin_events", checkin_df, overwrite=True)
    load_to_duckdb(conn, "presecurity_events", sec_df, overwrite=True)
    load_to_duckdb(conn, "boarding_events", boarding_df, overwrite=True)
    load_to_duckdb(conn, "disruption_events", disruption_df, overwrite=True)

    # Quick sanity counts
    for t in ["dim_airport", "flights", "checkin_events", "presecurity_events", "boarding_events", "disruption_events"]:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  - {t}: {cnt} rows")

    print("\n✅ DuckDB created at: data/amadeus_ops.duckdb")
    print("Next: we will build GOLD views in Step 3 (hourly KPI, top delay reason, anomaly scores).")

if __name__ == "__main__":
    main()