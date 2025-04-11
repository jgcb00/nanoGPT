module load GCCcore/.13.3.0 && module load Python/3.12.3 && module load NVHPC
python -m venv venv
source venv/bin/activate
pip install --force-reinstall --ignore-installed packaging rich requests tyro pyyaml transformers
pip install torch==2.5.0 torchvision torchaudio
python -m pip install torchrun_jsc
pip install ninja packaging psutil wheel
pip install flash-attn --no-build-isolation
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
pip install tqdm huggingface-hub wandb einops tiktoken
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"