# Sports Vision Model

This directory contains the source code for inference using Hooper's Sports Vision Model, which detects and tracks objects and players in the context of basketball.

### Docker Installation

Clone the repo:
```
git clone https://github.com/hooper-ai/hooper-ai.git
```
Run the setup script in `docker/`:
```
chmod +x setup.sh
./setup.sh
```
Start the container:
```
docker-compose up -d
```
Enter the container:
```
docker-compose exec hooper-ai bash
```

### Manual Installation

On a fresh runpod, from the home directory, run:
```
bash runpod.sh
```
This will install things like `git-lfs` that we need.
Then clone the repo:
```
git clone https://github.com/hooper-ai/hooper-ai.git
```

We use a `conda` environment with CUDA 12.2. Consider the following steps:
```
# Create a conda environment
conda create -n hooper-ai python=3.10.13
conda activate hooper-ai

# Install requirements
pip install -r requirements.txt

# Install Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```
