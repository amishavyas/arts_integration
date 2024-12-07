import csv
import random
import time
import string

def generate_random_row():
    pairID = 0
    subID = random.choice([0, 1])
    imgID = f"img_{random.randint(1, 12):02d}"
    text = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))  # Random string of length 5
    timestamp = time.time()

    return [pairID, subID, imgID, None, text, timestamp]


def is_audio_path_unique(audio_path, csv_filename):
    try:
        with open(csv_filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            existing_paths = set(row[3] for row in csv_reader if row)  # Column index 3 is audio_path
            return audio_path not in existing_paths
    except FileNotFoundError:
        return True  # File not found, audio path is unique

def add_rows_to_csv(num_rows):
    csv_filename = "session_output.csv"
    header = ["pairID", "subID", "imgID", "audio_path", "text", "timestamp"]

    # Generate new rows
    new_rows = []
    for _ in range(num_rows):
        # Generate a random 10-character string
        random_path = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        audio_path = f"{random_path}/{random.randint(1000, 9999)}.wav"

        while not is_audio_path_unique(audio_path, csv_filename):
            random_path = "".join(random.choices(string.ascii_letters + string.digits, k=10))
            audio_path = f"{random_path}/{random.randint(1000, 9999)}.wav"

        new_row = generate_random_row()
        new_row[3] = audio_path  # Set audio_path in the row
        new_rows.append(new_row)

    # Append rows to the CSV file
    with open(csv_filename, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            csv_writer.writerow(header)

        # Write the new rows
        csv_writer.writerows(new_rows)
 






