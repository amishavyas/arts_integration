import pandas as pd
import whisper
import os
import string
import torch
from tqdm import tqdm

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'='*50}")
print(f"Device: {device.upper()}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA not available, using CPU (this will be slower)")
print(f"{'='*50}\n")

# Load model on the appropriate device
print(f"Loading Whisper 'large' model on {device.upper()}...")
model = whisper.load_model("large", device=device)
print("Model loaded successfully!\n")

AUDIO_ROOT = "./data"

filler_words = ["yeah", "um", "uh", "okay", "ok", "hmm", "huh", "mm", "mmm", "huh", "you know", "like", "so", "right", "i mean", "well", "yes", "oh", "ah", "er", "ahh", "hmm", "hmmm", "uhh", "umm", "uhhh", "ummm"]
backchannels = ["yep","yup","ok","okay","alright","right","sure","exactly","uh huh","mm hmm","yuh","got it","i see","makes sense","uhuh","yessir"]
vague_responses = ["i don't know","idk","maybe","not sure","possibly","could be","whatever","nothing really","something like that","sort of","kind of","i guess","who knows","dunno","no idea","hard to say"]
# generic_responses = ["","cool","nice","neat","great","good","fine","awesome","wow","amazing","really","seriously","exactly","totally","definitely","for sure","looks good","sounds good","i see","makes sense","all right","sure thing",]
# greetings_etiquette_farewells = ["hi","hello","hey","good morning","good afternoon","good evening","how are you","how’s it going","hope you're well","thank you","thanks","thanks a lot","thanks so much","no problem","you're welcome","my pleasure","sorry","excuse me","pardon","goodbye","bye","see you","see ya","later","take care","have a good day","have a nice day","cheers"]
# random_words = ["next",]

# clean text
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    return text.translate(str.maketrans("", "", string.punctuation))

COMBINED_NOISE = set().union(filler_words, backchannels, vague_responses)

# filter
def should_remove(t: str) -> bool:
    if not t:
        return True
    if t in COMBINED_NOISE:
        return True
    words = t.split()
    return all(w in COMBINED_NOISE for w in words)

# detect rows that only contains numbers
def is_numbers_only(text: str) -> bool:
    s = str(text).strip().replace(" ", "")
    return bool(s) and s.isdigit()

# loop through folders

# Get all subfolders to process
subfolders = sorted(os.listdir(AUDIO_ROOT))
print(f"Found {len(subfolders)} subfolders to process\n")

for subfolder in tqdm(subfolders, desc="Processing folders", unit="folder"):
    subfolder_path = os.path.join(AUDIO_ROOT, subfolder)
    audio_dir = os.path.join(subfolder_path, "audio")
    csv_path = os.path.join(subfolder_path, "data.csv")

    # Check if already processed
    filtered_csv = os.path.join(subfolder_path, f"{subfolder}_transcribed_filtered.csv")
    removed_csv = os.path.join(subfolder_path, f"{subfolder}_transcribed_removed.csv")
    
    if os.path.isfile(filtered_csv) and os.path.isfile(removed_csv):
        tqdm.write(f"⏭️  Skipping {subfolder}: already processed")
        continue

    # Only process if both the audio folder and data.csv exist
    if not (os.path.isdir(audio_dir) and os.path.isfile(csv_path)):
        tqdm.write(f"⚠️  Skipping {subfolder}: missing audio/ or data.csv")
        continue

    # Try to read and validate CSV
    try:
        df = pd.read_csv(csv_path)
        if 'audio_path' not in df.columns:
            tqdm.write(f"❌ Skipping {subfolder}: CSV missing 'audio_path' column. Available columns: {list(df.columns)}")
            continue
    except Exception as e:
        tqdm.write(f"❌ Skipping {subfolder}: Error reading CSV - {str(e)}")
        continue

    # Collect all .wav files first to show progress
    wav_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append((root, file))
    
    if len(wav_files) == 0:
        tqdm.write(f"⚠️  Skipping {subfolder}: no .wav files found in audio directory")
        continue
    
    tqdm.write(f"\n📁 Processing folder: {subfolder} ({len(wav_files)} audio files)")
    
    transcription_dict = {}

    # Transcribe each audio file with progress bar
    for root, file in tqdm(wav_files, desc=f"  Transcribing {subfolder}", unit="file", leave=False):
        try:
            file_path = os.path.join(root, file)
            result = model.transcribe(file_path, language="en")
            transcription_dict[file] = result['text']
        except Exception as e:
            tqdm.write(f"⚠️  Failed to transcribe {file}: {str(e)}")
            transcription_dict[file] = ""  # Use empty string for failed transcriptions

    # Filter dataframe to only include files that were transcribed
    df = df[df["audio_path"].apply(lambda x: os.path.basename(x) in transcription_dict)]
    
    if len(df) == 0:
        tqdm.write(f"⚠️  Skipping {subfolder}: no matching audio files found in CSV")
        continue
    
    # Add transcriptions
    df["transcription"] = df["audio_path"].apply(lambda x: transcription_dict.get(os.path.basename(x), ""))

    # clean + mark removals
    df["cleaned_text"] = df["transcription"].apply(clean_text)
    df["to_remove"] = df["cleaned_text"].apply(should_remove) | df["cleaned_text"].apply(is_numbers_only)

    # split, drop helper columns, save both new csvs
    removed_df = df[df["to_remove"]].copy() # keep cleaned_text for auditing
    filtered_df = df[~df["to_remove"]].copy()
    filtered_df.drop(columns=["cleaned_text","to_remove"], inplace=True, errors="ignore")

    try:
        filtered_df.to_csv(filtered_csv, index=False)
        removed_df.to_csv(removed_csv, index=False)
        tqdm.write(f"✅ {subfolder}: {len(filtered_df)} kept, {len(removed_df)} removed")
    except Exception as e:
        tqdm.write(f"❌ Failed to save results for {subfolder}: {str(e)}")

print(f"\n{'='*50}")
print("✨ Processing complete!")
print(f"{'='*50}")