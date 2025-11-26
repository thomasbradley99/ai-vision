# Why It Didn't Work on Matteo's VM (and How to Fix It)

## Why It Failed Before

### 1. **Missing Re-ID Model Weights** (Most Likely Cause)
- The script needs: `swin_base_msmt17.pth` (336MB file)
- Location required: `hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth`
- **On this VM**: ✅ File exists at `/home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth`
- **On Matteo's VM**: ❌ Probably missing - script would error immediately

### 2. **Missing or Incomplete Conda Environment**
- Requires: `hooper-ai` conda environment
- Needs specific packages:
  - PyTorch (with CUDA support if GPU available)
  - Detectron2
  - Solider Re-ID model dependencies
  - OpenCV, NumPy, etc.
- **On this VM**: ✅ Environment exists and is configured
- **On Matteo's VM**: ❌ Might be missing or have wrong packages

### 3. **Wrong File Paths**
- Scripts assume specific directory structure
- If repo cloned to different location, paths break
- This VM: `/home/ubuntu/ai-vision/`
- Matteo's VM might have: `/home/ubuntu/clann/ai-vision/` or different path

### 4. **Missing Dependencies/Python Modules**
- Missing `hooper` module (from hooper-glean package)
- Missing Detectron2 installation
- Missing Solider dependencies

### 5. **GPU/Device Issues**
- Script defaults to `--device cuda`
- If no GPU or CUDA not configured → crashes
- Need to use `--device cpu` if no GPU

## How to Make It Work on Matteo's VM

### Step 1: Check What's Already There

```bash
# SSH into Matteo's VM
ssh user@matteo-vm

# Check if repo exists
ls -la /home/ubuntu/ai-vision
# or
ls -la /home/ubuntu/clann/ai-vision

# Check for Re-ID weights
find /home/ubuntu -name "swin_base_msmt17.pth" 2>/dev/null

# Check conda environment
conda env list | grep hooper
```

### Step 2: Copy Missing Files from This VM

#### Option A: Copy Re-ID Weights (336MB)
```bash
# From THIS VM (current working one):
cd /home/ubuntu/ai-vision
scp hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth user@matteo-vm:/path/to/destination/

# On Matteo's VM, place it:
mkdir -p /home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking
mv swin_base_msmt17.pth /home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/
```

#### Option B: Use rsync (Better for multiple files)
```bash
# From THIS VM:
rsync -avz --progress \
    hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth \
    user@matteo-vm:/home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/
```

### Step 3: Export Conda Environment

```bash
# On THIS VM (working one):
conda activate hooper-ai
conda env export -n hooper-ai > hooper-ai-env.yml

# Copy environment file to Matteo's VM
scp hooper-ai-env.yml user@matteo-vm:/home/ubuntu/
```

### Step 4: Setup on Matteo's VM

```bash
# SSH into Matteo's VM
ssh user@matteo-vm

# 1. Clone/update repo (if not already there)
cd /home/ubuntu
git clone <repo-url> ai-vision
# or update existing:
cd ai-vision && git pull

# 2. Create conda environment
conda env create -f hooper-ai-env.yml
# Or if it exists but is broken:
conda env remove -n hooper-ai
conda env create -f hooper-ai-env.yml

# 3. Activate environment
conda activate hooper-ai

# 4. Verify Re-ID weights exist
ls -lh /home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth
# Should show ~336MB file

# 5. Test imports
python -c "import torch; import detectron2; print('Detectron2 OK')"
python -c "from hooper.solider_utils import get_solider_feature_extractor; print('Solider OK')"
```

### Step 5: Test the Script

```bash
# On Matteo's VM:
cd /home/ubuntu/ai-vision/gaa-vision-ai

# Test with help command (should work without errors)
conda activate hooper-ai
python scripts/track_individual_improved.py --help

# Test with CPU first (safer)
python scripts/track_individual_improved.py \
    --video inputs/test_2s.mp4 \
    --reference-image inputs/reference_frame.jpg \
    --output outputs/test.mp4 \
    --device cpu \
    --frame-stride 2
```

### Step 6: Check GPU (Optional)

```bash
# Check if GPU available
nvidia-smi

# Check if PyTorch sees GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If GPU available, use --device cuda
# If not, use --device cpu (slower but works)
```

## Quick Checklist for Matteo's VM

- [ ] Repo cloned/updated: `/home/ubuntu/ai-vision/`
- [ ] Re-ID weights exist: `hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth` (336MB)
- [ ] Conda environment: `hooper-ai` exists
- [ ] Dependencies installed: PyTorch, Detectron2, hooper module
- [ ] Script path correct: `/home/ubuntu/ai-vision/gaa-vision-ai/scripts/track_individual_improved.py`
- [ ] Test run works: `python scripts/track_individual_improved.py --help`

## Common Errors and Fixes

### Error: "Re-ID weights not found"
**Fix**: Copy `swin_base_msmt17.pth` to correct location

### Error: "No module named 'hooper'"
**Fix**: Need to install hooper-glean package or add to Python path

### Error: "CUDA out of memory" or "NVIDIA driver not found"
**Fix**: Use `--device cpu` instead of `--device cuda`

### Error: "No such file or directory" for script
**Fix**: Wrong working directory - use correct path structure

## Recommended Setup Script

Create this on Matteo's VM:

```bash
#!/bin/bash
# setup_matteo_vm.sh

# 1. Ensure repo is at correct location
cd /home/ubuntu
if [ ! -d "ai-vision" ]; then
    git clone <repo-url> ai-vision
fi

# 2. Ensure checkpoints directory exists
mkdir -p ai-vision/hooper-glean/checkpoints/PERSON-Tracking

# 3. Check if weights exist
if [ ! -f "ai-vision/hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth" ]; then
    echo "ERROR: Re-ID weights missing!"
    echo "Copy from working VM:"
    echo "  scp /path/to/swin_base_msmt17.pth user@matteo-vm:/home/ubuntu/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/"
    exit 1
fi

# 4. Activate conda environment
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate hooper-ai

# 5. Test
cd ai-vision/gaa-vision-ai
python scripts/track_individual_improved.py --help

echo "Setup complete!"
```

## Summary

**Why it didn't work**: Most likely missing the 336MB Re-ID model weights file or incomplete conda environment.

**How to fix**: 
1. Copy `swin_base_msmt17.pth` from this VM to Matteo's VM
2. Ensure conda environment is set up correctly
3. Verify all paths are correct
4. Test with `--device cpu` first, then GPU if available

The key difference: This VM has the model weights file, Matteo's VM probably doesn't.

