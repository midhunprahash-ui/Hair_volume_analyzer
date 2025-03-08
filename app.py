from flask import Flask, render_template
import mysql.connector

app = Flask(__name__)

# Function to connect to MySQL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Change if you've set a MySQL password
        password="",  # Set your MySQL password if needed
        database="hair_analysis"
    )


def get_hair_data():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT date, time, volume FROM hair_data ORDER BY id DESC LIMIT 5")
    data = cursor.fetchall()
    
    conn.close()
    return data

#
def analyze_trend():
    data = get_hair_data()

    if len(data) < 2:
        return "Not enough data for trend analysis."

    volumes = [entry['volume'] for entry in data]

    if volumes[0] < min(volumes[1:]):  # Compare latest volume with previous ones
        return "You have a gradual hair loss, so better take some precautions."
    else:
        return "You have significant hair growth, keep it up!"

@app.route('/')
def index():
    data = get_hair_data()
    trend_message = analyze_trend()
    return render_template("index.html", data=data, trend_message=trend_message)

if __name__ == '__main__':
    app.run(debug=True)
    if else
    