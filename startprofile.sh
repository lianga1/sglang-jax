conda deactivate
source ./.venv/bin/activate
pip install transformers 
uv pip install libtpu
uv pip install jaxtyping
chmod a+x ./hfd.sh

alias hfd='./hfd.sh'

sudo apt install aria2 -y

hfd inclusionAI/Ling-mini-2.0 --local-dir inclusionAI/Ling-mini-2.0

git config --global user.email "1838169875@qq.com"
git config --global user.name "liangzuning"
git checkout ling_minimal

git clone https://github.com/AI-Hypercomputer/maxtext.git
git clone https://github.com/jax-ml/bonsai.git