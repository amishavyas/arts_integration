# %%
# Make a reverse-reverb version of every segment and layer it under the clean segment.
# For each segment: cut it from its track using start/end, reverse it, add a reverb
# tail of --tail_s seconds, reverse again (so the reverb swells up into the speech),
# then add the unprocessed segment on top, placed so it lines up sample-exactly with
# the audio the reverb was made from. Once the speech starts the reverb is turned
# down (--duck) so the speech is clear with some reverb still under it. Each output is tail_s + segment long, and the
# clean speech starts tail_s seconds in (see index.csv).
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from argparse import ArgumentParser
from pedalboard import Reverb

from paths import DATA_ROOT

# %%
args = ArgumentParser()
args.add_argument('--tail_s', type=float, default=4, help='length of the reverb swell before each segment')
args.add_argument('--room_size', type=float, default=0.9, help='0.9 decays to -60 dB in about 4 s')
args.add_argument('--damping', type=float, default=0.5)
args.add_argument('--wet_db', type=float, default=-6, help='reverb level relative to the clean segment')
args.add_argument('--duck', type=float, default=0.5, help='reverb gain once the clean speech starts (1 = no change)')
args.add_argument('--duck_ms', type=float, default=50, help='length of the ramp down, starting at the speech onset')
args.add_argument('--limit', type=int, default=None, help='only process the first N segments (for testing)')
args.add_argument('--out_dir', type=Path, default=None)

args = args.parse_args()
print(args)
ddir = DATA_ROOT / 'arts_integration_segments_audio'
out_dir = args.out_dir or ddir / 'solo_fx'
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ddir / 'aggregated_segments_with_embeddings.csv',
                 usecols=['audio_path', 'start', 'end'], nrows=args.limit)

# %%
reverb = Reverb(room_size=args.room_size, damping=args.damping,
                wet_level=1.0, dry_level=0.0, width=1.0)
wet_gain = 10 ** (args.wet_db / 20)

rows = []
n_scaled = 0
for audio_path, group in df.groupby('audio_path'):
    track, sr = sf.read(ddir / audio_path, dtype='float32')
    pad = int(args.tail_s * sr)
    for row in group.itertuples():
        seg = track[int(row.start * sr):int(row.end * sr)]
        # Reverse, give the reverb room to ring out, reverb only (no dry), reverse back
        rev = np.concatenate([seg[::-1], np.zeros(pad, np.float32)])
        wet = reverb(rev[np.newaxis, :], sr, reset=True)[0][::-1]
        # The reversed-back input is the clean segment starting at `pad`
        dry = np.concatenate([np.zeros(pad, np.float32), seg])
        # Full reverb for the swell, then ramp down to --duck once the speech starts
        env = np.full(len(wet), args.duck, np.float32)
        env[:pad] = 1
        ramp = env[pad:pad + int(args.duck_ms / 1000 * sr)]
        ramp[:] = np.linspace(1, args.duck, len(ramp), dtype=np.float32)
        out = dry + wet_gain * env * wet

        # Only scale down if the mix would clip
        peak = np.abs(out).max()
        if peak > 1:
            out /= peak
            n_scaled += 1

        fx_file = f'{Path(audio_path).stem}_{int(round(row.start * 1000)):08d}-{int(round(row.end * 1000)):08d}.wav'
        sf.write(out_dir / fx_file, out, sr, subtype='PCM_16')
        rows.append({'audio_path': audio_path, 'start': row.start, 'end': row.end,
                     'fx_file': fx_file, 'clean_onset_s': pad / sr})
    print(f'{audio_path}: {len(group)} segments')

pd.DataFrame(rows).to_csv(out_dir / 'index.csv', index=False)
print(f'Wrote {len(rows)} files to {out_dir} ({n_scaled} scaled down to avoid clipping)')
