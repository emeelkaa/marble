# 🧠 MARBLE [MICCAI 2026]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of our MICCAI 2026 paper: **MARBLE**: Lightweight EEG-to-fMRI Translation via Mamba-Attention and ROI-Conditioned Decoding.
The study addresses the challenging task of EEG-to-fMRI synthesis by introducing a novel Mamba-Attention architecture and an ROI-conditioned cross-attention mechanism for efficient and interpretable multimodal brain signal translation. 
The proposed framework achieves strong reconstruction performance while substantially improving computational efficiency, reducing the number of model parameters by 3.2× and peak memory consumption by 2.3× compared with existing approaches.

## 📘 Overview
![Framework Overview](pipeline.png)
---

## 📂 Repository Structure
```
marble/
├── models/            
│   ├── marble.py           # Model
├── main.py                 # Main training script
├── engine.py               # Training engine
├── optim.py                # Optimizer 
├── utils.py                # Utilities (e.g., metric logging)
├── requirements.txt        # Python dependencies
├── README.md               # This file
```
### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/emeelkaa/marble.git
   cd marble
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch** (CUDA 12.1)
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

4. **Install Mamba**

Follow the official instructions at [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba). Both `mamba-ssm` and `causal-conv1d` must be built for your CUDA version.

5. **Install remaining dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Download the datasets**
   - https://huggingface.co/ssssssup/NeuroBOLT
  
2. Please refer to the Neurobolt authors for preprocessing. Run it 
https://github.com/soupeeli/NeuroBOLT

3. **Train the model**:
```bash
python main.py \
    --prepro_datapath /your/path/to/vu.pkl \
    --batch_size 64 \
    --epochs 30 \
    --emb_size 128 \
    --depth 2 \
    --train_test_mode full_test \
    --output_dir ./checkpoints
```

4. **Evaluation only**:
```bash
python main.py \
    --prepro_datapath /your/path/to/vu.pkl \
    --train_test_mode full_test \
    --output_dir ./checkpoints \
    --eval
```

## TODO

- [x] Release code
- [ ] Upload Colab notebook for demonstration
- [ ] Release checkpoints

## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

- **Email**: [emilkim01@pusan.ac.kr](mailto:emilkim01@pusan.ac.kr)
- **Author**: Emil Kim

---
