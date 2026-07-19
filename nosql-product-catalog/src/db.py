"""MongoDB connection helper.

Reads the connection string from the MONGO_URI environment variable so
the same code runs against a local docker-compose Mongo, a CI service
container, or a managed cluster (Atlas) with no code change. Defaults to
localhost.
"""

from __future__ import annotations

import os

from pymongo import MongoClient

DB_NAME = "catalog"


def get_client() -> MongoClient:
    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    # serverSelectionTimeoutMS keeps failures fast (5 s) instead of the
    # 30 s default -- matters for the "skip tests if no Mongo" path.
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_db(client: MongoClient | None = None):
    return (client or get_client())[DB_NAME]
