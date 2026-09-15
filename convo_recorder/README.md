## Environment Setup

1. Install Miniconda or Anaconda

2. Create and activate the environment:
```bash
conda create -n art-recorder python=3.8 nodejs
conda activate art-recorder
```

3. Install Python dependencies:
```bash
conda install -c conda-forge flask flask-cors sounddevice numpy pandas scipy librosa
pip install openai-whisper torch
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