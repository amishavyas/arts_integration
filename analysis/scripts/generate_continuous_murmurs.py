# %%
# Generate continuous murmurs by concatenating speech segments across all sessions.
# Each segment is cut from its speaker's isolated track (audio_path) using the
# start/end times in the CSV, so only the segmented speech is used, never the gaps.
import pandas as pd
from scipy.io import wavfile
import numpy as np
from argparse import ArgumentParser

from paths import DATA_ROOT

# %%
args = ArgumentParser()
args.add_argument('--n_outputs', type=int, default=1)
args.add_argument('--length_minutes', type=float, default=10)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--fade_ms', type=float, default=5, help='fade in/out on each segment to avoid clicks; 0 disables')

args = args.parse_args()
print(args)
ddir = DATA_ROOT / 'arts_integration_segments_audio'
df = pd.read_csv(ddir / 'aggregated_segments_with_embeddings.csv',
                 usecols=['audio_path', 'start', 'end'])

# %%
# Load each track once, then cut out only the segment portions
sources = {p: wavfile.read(ddir / p) for p in df['audio_path'].unique()}
srs = {sr for sr, _ in sources.values()}
assert len(srs) == 1, f'mixed sample rates: {srs}'
sr = srs.pop()

segments = []
for row in df.itertuples():
    data = sources[row.audio_path][1]
    seg = data[int(row.start * sr):int(row.end * sr)].astype(np.float32)
    n_fade = min(int(args.fade_ms / 1000 * sr), len(seg) // 2)
    if n_fade > 0:
        ramp = np.linspace(0, 1, n_fade, dtype=np.float32)
        seg[:n_fade] *= ramp
        seg[-n_fade:] *= ramp[::-1]
    segments.append(seg.astype(data.dtype))
del sources
print(f'{len(segments)} segments, {sum(map(len, segments)) / sr / 60:.1f} min of speech total')

# %%
rng = np.random.default_rng(args.seed)
target = int(args.length_minutes * 60 * sr)
for i in range(args.n_outputs):
    # Randomly sample segments until the target length is reached
    chunks, total = [], 0
    while total < target:
        seg = segments[rng.integers(len(segments))]
        chunks.append(seg)
        total += len(seg)
    concatenated_audio = np.concatenate(chunks)[:target]
    # Write the concatenated audio to a new file
    output_filename = ddir / f'continuous_murmur_{i}.wav'
    wavfile.write(output_filename, sr, concatenated_audio)
    print(f'Generated {output_filename}')
