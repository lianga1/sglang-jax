conda deactivate
source ./.venv/bin/activate
pip install transformers 
uv pip install libtpu
huggingface-cli download inclusionAI/Ling-mini-2.0 --local-dir inclusionAI/Ling-mini-2.0

git config --global user.email "1838169875@qq.com"
git config --global user.name "liangzuning"
git checkout ling_minimal
