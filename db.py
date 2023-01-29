import sqlite3
import random

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

items = ['adidas_cap', 'american_crew_hair_paste', 'coconut_water', 'oven_mitt',
         'pink_ukelele', 'scissors', 'spoon', 'teddy_hamster', 'water_bottle', 'computer_mouse',
         'disinfecting_cleaner', 'febreeze_spray', 'fork', 'keyboard', 'knife', 'ladle', 'nutella',
         'omachi_mi_bap_bo', 'omachi_sot_bo_ham', 'phone', 'shear_revival_clay_pomade', 'skippy_peanut_butter',
         'sprite', 'teddy_octopus', 'watch', 'wine']

for item in items:
    price = random.randint(2, 50)
    cursor.execute("INSERT INTO Items (item, price) VALUES (?, ?)",
                    (item, price))
    conn.commit()
conn.close()