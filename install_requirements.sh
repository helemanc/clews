# We assume python>=3.10

# --- Create environment and activate ---
python3 -m venv .venv
source .venv/bin/activate

# --- Pytorch ---
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.3.0 tensorboard==2.17.0 einops==0.8.0 torchinfo==1.8.0

# --- Generic stuff ---
pip install omegaconf==2.3.0 tqdm==4.66.4

# --- For metadata proc ---
pip install joblib==1.4.2

# --- For my audio_utils ---
pip install soundfile==0.12.1 soxr==0.3.7 nnAudio==0.3.3
# Need to downgrade numpy for soxr
pip install numpy==1.26.4
# ffmpeg needs to be installed. Can do it through conda if you do not have sudo privileges:
conda install -c conda-forge 'ffmpeg<7'
# Alternatively, use:
#sudo apt install ffmpeg

# --- For data augmentation (may be imported but unused) ---
pip install julius==0.2.7

# --- For text processing ---
pip install nltk
