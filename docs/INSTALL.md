# Installation

### Acknowledgement: This readme file for installing datasets is modified from [PromptKD](https://github.com/zhengli97/PromptKD/tree/main)'s official repository.

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

**Note**: We have made many modifications to the original Dassl runtime. Therefore, please use the code in this repository to initialize Dassl.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n dpc python=3.8

# Activate the environment
conda activate dpc

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Clone PromptKD code repository and install requirements
```bash
# Clone PromptSRC code base
git clone https://github.com/JREion/DPC.git

cd DPC/
# Install requirements

pip install -r requirements.txt

cd ..
```

* Install Dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
# original source: https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
