## Environment Setup

1. Install Miniconda or Anaconda

2. Create and activate the environment:
```bash
conda create -n art-recorder python=3.8 nodejs
conda activate art-recorder
```

3. Install Python dependencies (there are probably more than those listed here):
```bash
conda install -c conda-forge numpy pandas scipy
pip install openai-whisper torch sounddevice librosa flask flask-cors psutils
```

4. Install frontend dependencies:
```bash
cd frontend
npm install
```

5. Run the application:
```bash
python run.py
``` 