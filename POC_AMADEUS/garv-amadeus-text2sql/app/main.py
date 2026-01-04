from __future__ import annotations

import argparse
import json
from dotenv import load_dotenv

from app.graph.text2sql_graph import run_text2sql


def run_cli(question: str) -> None:
    out = run_text2sql(question)
    print("\n=== FINAL SQL ===")
    print(out.get("final_sql"))
    print("\n=== PREVIEW ===")
    print(out.get("preview_markdown"))
    print("\n=== EXPLANATION ===")
    print(json.dumps(out.get("explanation"), indent=2, default=str))


def build_api():
    from fastapi import FastAPI
    from app.api.routes import router

    api = FastAPI(title="GARV Text2SQL API", version="0.1")

    @api.get("/")
    def root():
        return {
            "message": "GARV Text2SQL API is running",
            "endpoints": {
                "health": "/api/health",
                "text2sql": "/api/text2sql",
                "docs": "/docs",
                "openapi": "/openapi.json",
            },
        }

    api.include_router(router, prefix="/api")
    return api


def run_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    api = build_api()
    uvicorn.run(api, host=host, port=port, log_level="info")


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--api", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.api:
        run_api(host=args.host, port=args.port)
        return

    if not args.question:
        print("Provide a question or run with --api")
        print("Example:")
        print('  python -m app.main "Top 5 airports by avg security wait time last 7 days"')
        print("  python -m app.main --api --host 127.0.0.1 --port 8000")
        return

    run_cli(args.question)


if __name__ == "__main__":
    main()