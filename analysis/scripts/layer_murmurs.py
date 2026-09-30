# %%
# Layer continuous murmurs (from generate_continuous_murmurs.py) on top of one another.
import numpy as np
from scipy.io import wavfile
from argparse import ArgumentParser

from paths import DATA_ROOT

# %%
args = ArgumentParser()
args.add_argument('--n_layers', type=int, default=2)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--peak_dbfs', type=float, default=-1, help='peak level of the mix after normalizing')

args = args.parse_args()
print(args)
ddir = DATA_ROOT / 'arts_integration_segments_audio'
files = sorted(ddir.glob('continuous_murmur_*.wav'))
if args.n_layers > len(files):
    raise SystemExit(f'--n_layers {args.n_layers} but only {len(files)} murmur files in {ddir}; '
                     f'run generate_continuous_murmurs.py with --n_outputs {args.n_layers} first')

# %%
# Pick distinct murmurs and sum them, trimming to the shortest
rng = np.random.default_rng(args.seed)
chosen = rng.choice(files, size=args.n_layers, replace=False)
layers = []
for f in chosen:
    sr, data = wavfile.read(f)
    layers.append((sr, data))
    print(f'Layering {f.name}')
srs = {sr for sr, _ in layers}
assert len(srs) == 1, f'mixed sample rates: {srs}'
sr = srs.pop()
n = min(len(data) for _, data in layers)
mix = np.sum([data[:n].astype(np.float32) for _, data in layers], axis=0)

# Normalize so the summed layers don't clip
peak = np.abs(mix).max()
if peak > 0:
    mix *= (10 ** (args.peak_dbfs / 20) * np.iinfo(np.int16).max) / peak

# %%
output_filename = ddir / f'layered_murmur_{args.n_layers}layers.wav'
wavfile.write(output_filename, sr, mix.astype(np.int16))
print(f'Generated {output_filename} ({n / sr / 60:.1f} min)')
