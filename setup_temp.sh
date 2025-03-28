pip3 install --upgrade torch torchvision torchaudio
pip3 install -U git+https://github.com/fla-org/flash-linear-attention
pip install flash-attn --no-build-isolation
git config --global user.name "alxndrTL" && git config --global user.email "alextorresleguet@icloud.com"
pip install transformers pandas matplotlib tyro wandb tiktoken
wandb login 6677ca4ea7a459489b4dbbcd2503acd8e7de27d2
git clone https://github.com/XunhaoLai/native-sparse-attention-triton
mv native-sparse-attention-triton/native_sparse_attention/ .
rm -rf native-sparse-attention-triton/