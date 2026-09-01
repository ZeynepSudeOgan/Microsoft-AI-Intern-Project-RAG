import sqlite3
import json

DB_PATH = "rag.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_document(content, embedding):
    conn = get_connection()
    conn.execute(
        "INSERT INTO documents (content, embedding) VALUES (?, ?)",
        (content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def get_all_documents():
    conn = get_connection()
    rows = conn.execute("SELECT id, content, embedding FROM documents").fetchall()
    conn.close()
    return [
        {"id": row[0], "content": row[1], "embedding": json.loads(row[2])}
        for row in rows
    ]


if __name__ == "__main__":
    init_db()
    print("Database initialized: rag.db (table: documents)")