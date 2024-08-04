# on gpu node
pip install torch torchaudio torchvision
pip install packaging
pip install wheel setuptools
pip install huggingface_hub
pip install opencv-python-headless
pip install numpy==1.26.3
pip install transformers==4.37.2 peft==0.10.0 gradio decord einops moviepy imageio openai
# on login node
pip install flash-attn --no-build-isolation