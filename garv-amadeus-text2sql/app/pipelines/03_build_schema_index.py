from __future__ import annotations

from dotenv import load_dotenv

from app.rag.schema_index import build_schema_index


def main() -> None:
    load_dotenv(override=True)

    # Optional: only index gold/silver tables
    # table_prefixes = ["gold_", "silver_"]
    table_prefixes = None

    build_schema_index(force_rebuild=True, table_prefixes=table_prefixes)
    print("✅ Schema index rebuilt and persisted to data/chroma_schema_index")


if __name__ == "__main__":
    main()