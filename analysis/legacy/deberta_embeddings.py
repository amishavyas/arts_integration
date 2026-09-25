import pandas as pd
import os
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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

# Load model
print(f"Loading DeBERTa-v3-base model on {device.upper()}...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
model = model.to(device)
print("Model loaded successfully!\n")

# Define mean pooling function
def mean_pooling(model_output, attention_mask):
    """
    Create sentence embeddings by averaging token embeddings.
    Uses attention_mask to exclude padding tokens.
    This is the recommended approach for semantic clustering.
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    # Expand attention_mask to match the shape of token_embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum the token embeddings, ignoring padded tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum the attention masks to get the number of non-padded tokens
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Calculate the mean
    return sum_embeddings / sum_mask

AUDIO_ROOT = "./data"
FILTERED_CSV = "_transcribed_filtered.csv"
EMBEDDING_FILE = "_embeddings.pt"

# Get all subfolders to process
subfolders = sorted(os.listdir(AUDIO_ROOT))
print(f"Found {len(subfolders)} subfolders to process\n")

for subfolder in tqdm(subfolders, desc="Processing folders", unit="folder"):
    subfolder_path = os.path.join(AUDIO_ROOT, subfolder)
    
    # Check if filtered CSV exists
    filtered_csv = os.path.join(subfolder_path, f"{subfolder}{FILTERED_CSV}")
    
    if not os.path.isfile(filtered_csv):
        tqdm.write(f"⚠️  Skipping {subfolder}: filtered CSV not found")
        continue
    
    # Check if already processed
    embedding_path = os.path.join(subfolder_path, f"{subfolder}{EMBEDDING_FILE}")
    
    if os.path.isfile(embedding_path):
        tqdm.write(f"⏭️  Skipping {subfolder}: embeddings already generated")
        continue
    
    # Try to read and validate CSV
    try:
        df = pd.read_csv(filtered_csv)
        if 'transcription' not in df.columns:
            tqdm.write(f"❌ Skipping {subfolder}: CSV missing 'transcription' column. Available columns: {list(df.columns)}")
            continue
    except Exception as e:
        tqdm.write(f"❌ Skipping {subfolder}: Error reading CSV - {str(e)}")
        continue
    
    if len(df) == 0:
        tqdm.write(f"⚠️  Skipping {subfolder}: empty CSV")
        continue
    
    tqdm.write(f"\n📁 Processing folder: {subfolder} ({len(df)} transcriptions)")
    
    all_embeddings = []
    max_length = 512
    
    # Process each transcription
    for index, row in tqdm(df.iterrows(), desc=f"  Generating embeddings for {subfolder}", total=len(df), unit="row", leave=False):
        try:
            line = row['transcription']
            
            # Skip empty transcriptions
            if pd.isna(line) or not str(line).strip():
                all_embeddings.append(None)
                continue
            
            # Tokenize with padding and truncation
            tokens = tokenizer(
                str(line).strip(), 
                return_tensors='pt', 
                truncation=True, 
                padding='max_length', 
                max_length=max_length
            )
            
            # Move to device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Get embeddings
            with torch.no_grad():
                model_output = model(**tokens)
            
            # Apply mean pooling to create sentence embedding
            embedding = mean_pooling(model_output, tokens['attention_mask'])
            
            # Optional: normalize embeddings for cosine similarity clustering
            # embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            all_embeddings.append(embedding)
            
        except Exception as e:
            tqdm.write(f"⚠️  Failed to process row {index+1}: {str(e)}")
            all_embeddings.append(None)
    
    # Filter out None values and concatenate
    valid_embeddings = [e for e in all_embeddings if e is not None]
    
    if len(valid_embeddings) == 0:
        tqdm.write(f"⚠️  Skipping {subfolder}: no valid embeddings generated")
        continue
    
    # Concatenate all embeddings
    all_embeddings_tensor = torch.cat(valid_embeddings, dim=0)
    
    # Save embeddings
    try:
        torch.save(all_embeddings_tensor, embedding_path)
        tqdm.write(f"✅ {subfolder}: Generated {len(valid_embeddings)} embeddings (shape: {all_embeddings_tensor.shape})")
    except Exception as e:
        tqdm.write(f"❌ Failed to save embeddings for {subfolder}: {str(e)}")

print(f"\n{'='*50}")
print("✨ Processing complete!")
print(f"{'='*50}")
