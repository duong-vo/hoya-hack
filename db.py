import sqlite3


def connection():
    try:
        conn = sqlite3.connect('item.db')
    except sqlite3.error as e:
        print("error", e)
    return conn

conn = connection()
cursor = conn.cursor()

# Create Terns table
cursor.execute("""CREATE TABLE IF NOT EXISTS Items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item VARCHAR(255) NOT NULL,
                    price INTEGER
                    );""")

conn.close()