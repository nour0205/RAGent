DOC_REGISTRY = {
    "postgres": {
        "document_id": "db_postgres",
        "description": "PostgreSQL MVCC and concurrency model",
        "aliases": ["postgresql", "postgres", "pg"]
    },
    "sqlserver": {
        "document_id": "db_sqlserver",
        "description": "SQL Server snapshot isolation",
        "aliases": ["sqlserver", "sql server", "sql_server", "mssql"]
    },
    "mongodb": {
        "document_id": "db_mongodb_indexing",
        "description": "MongoDB indexing and query performance",
        "aliases": ["mongodb", "mongo", "mongo db"]
    }
}

def normalize_document_id(value: str) -> str:
    if not value:
        return value

    key = value.strip().lower()

    for _, entry in DOC_REGISTRY.items():
        document_id = entry["document_id"]
        aliases = entry.get("aliases", [])

        if key == document_id.lower():
            return document_id

        if key in [alias.lower() for alias in aliases]:
            return document_id

    return value
