from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
import time
import datetime
import asyncio
import pandas as pd
from collections import deque
import threading
import csv
import random
from temp import add_rows_to_csv

# source dev_env/bin/activate
# flask --app server --debug run

app = Flask(__name__)

# Define variables for connection components
db_dialect = "mysql"
db_driver = "pymysql"
db_username = "root"
db_password = "adv2201"
db_host = "localhost"
db_port = 3306
db_name = "artsinteg"

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"{db_dialect}+{db_driver}://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db = SQLAlchemy(app)

# Define tables and data types

class TestDB(db.Model):
    pairID = db.Column(db.String(200))
    subID = db.Column(db.String(200))
    imgID = db.Column(db.String(10), nullable=False)
    audio_path = db.Column(db.String(500), primary_key=True)
    text = db.Column(db.String(500))
    timestamp = db.Column(db.String(500))
    embedding = db.Column(db.PickleType)


class PermDB(db.Model):
    pairID = db.Column(db.String(200))
    subID = db.Column(db.String(200))
    imgID = db.Column(db.String(10), nullable=False)
    audio_path = db.Column(db.String(500), primary_key=True)
    text = db.Column(db.String(500))
    timestamp = db.Column(db.String(500))
    embedding = db.Column(db.PickleType)


class RowTracker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    row = db.Column(db.Integer)

# Create tables
with app.app_context():
    db.create_all()

    # Check if the RowTracker is empty
    if not RowTracker.query.first():
        # Insert initial values
        db.session.add(RowTracker(id=1, row=0))  
        db.session.commit()

# Writes the current image ID in "img_data.txt"
@app.route("/write_img_data", methods=["GET", "POST"])
def update_file():
    data = request.get_json()
    with open("img_data.txt", "w") as file:
        file.write(data["img"])
    return "Current image updated successfully"

# Clears "img_data.txt" file
@app.route("/clear_file", methods=["GET"])
def clear_file():
    with open("img_data.txt", "w") as file:
        file.truncate(0)
    return "File wiped successfully"

# Write a row to temp_db
@app.route("/write_to_temp_db", methods=["GET", "POST"])
def add_record(data):
    # data is assumed to be a single row of values as a list
    record = TestDB(
        pairID=data[0],
        subID=data[1],
        imgID=data[2],
        audio_path=data[3],
        text=data[4],
        timestamp=data[5],
        embedding=[1, 2, 3, 4, 5],
    )

    db.session.add(record)
    db.session.commit()
    return "Data recorded in temp DB"


@app.route("/read_db", methods=["GET", "POST"])
def read_db():
    # reads the entire mySQL table and passes it on as a pd dataframe
    results = TestDB.query.all()
    pairID, subID, imgID, audio_path, embeddings = zip(
        *[(result.pairID, result.subID, result.imgID, result.audio_path, result.embedding) for result in results]
    )
    test_df = pd.DataFrame(
        data={"pairID": pairID, "subID": subID, "imgID": imgID, "audio_path": audio_path, "embedding": embeddings}
    )
    return test_df


@app.route("/read_test_db", methods=["GET"])
def get_all_rows():
    try:
        # Query all rows from the table
        rows = TestDB.query.all()

        # Convert the query result to a list of dictionaries
        result = [
            {
                "pairID": row.pairID,
                "subID": row.subID,
                "imgID": row.imgID,
                "audio_path": row.audio_path,
                "text": row.text,
                "embeddings": row.embeddings,
                "timestamp": row.timestamp,
            }
            for row in rows
        ]

        # Convert the result to a DataFrame
        df = pd.DataFrame(result)
        return df

    except Exception as e:
        return jsonify({"error": str(e)})


# Checks "session_output.csv" for new rows and updates temp_db 
@app.route("/transfer_utterance_data", methods=["GET", "POST"])
def transfer_utterance_data():
    currentRow = RowTracker.query.filter(RowTracker.id == 1).first()
    currentRowNumber = currentRow.row

    with open("session_output.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        rows_to_read = list(csv_reader)
        newRowNumber = len(rows_to_read)

        #print("total rows in csv", len(rows_to_read), )

        # Add new rows  
        for row in rows_to_read[currentRowNumber:]:
            with app.app_context():
                print("num of new rows", len(rows_to_read[currentRowNumber:]))
                add_record(row)

    currentRow.row = newRowNumber
    db.session.commit()

    return "new rows successfully added"


def async_worker():
    while True:
        with app.app_context():
            transfer_utterance_data()

# Below code adds random data to session_output.csv for testing purposes
def add_rows():
    while True:
        # Add a random number (1,10) of fake rows to session_output.csv every 2 seconds
        add_rows_to_csv(random.randint(1, 10))
        time.sleep(2)


threading.Thread(target=async_worker).start()
threading.Thread(target=add_rows).start()


if __name__ == "__main__":
    app.run(debug=False)
