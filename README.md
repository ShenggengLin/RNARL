
# RNARL: RNA Design via Reinforcement Learning

A PyTorch implementation of reinforcement learning-based RNA design with MoE architecture and ESM-2 integration.
![image](https://github.com/ShenggengLin/RNARL/blob/main/pictures/logo.png)

## Features

- 🧬 Transformer-based architecture with Mixture-of-Experts (MoE)
- 🔥 Distributed training support
- 🤖 Integrated with ESM-2 protein language model
- 📈 GRPO (Group-wise Relative Policy Optimization) algorithm
- 🚀 Multi-GPU training support



## Installation

1. Clone repository:
```bash
git clone https://github.com/ShenggengLin/RNARL.git
cd RNARL
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate RNARL
```

3. Download ESM-2 model:
```bash
mkdir esm2_model_t33_650M_UR50D

Download the esm2_model_t33_650M model weights and corresponding tokenizer to this folder.
```

## Project Structure
```
RNARL/
├── example_data               # The example datasets are in this folder
	├── example_train.csv
	├── example_val.csv
	└── example_test.csv
├── scripts               # The code files are in this folder
	├── actor_train.py          # Actor model training
	├── reward_train.py         # Reward model training
	├── grpo_train.py           # GRPO policy optimization
	├── transformer_encoder_MoE.py  # MoE transformer implementation
	├── utils.py                # Dataset and model utilities
├── pictures               
├── environment.yml         # Conda environment configuration
└── README.md
```



## Script Descriptions

### 1. Actor Model Training (`actor_train.py`)

**Function**:  
Trains a protein-to-mRNA translation model using:
- Transformer with Mixture-of-Experts (MoE) architecture
- ESM-2 protein language model integration
- Distributed training support

**Key Features**:
- Input: Protein sequences
- Output: mRNA sequence logits with codon constraints
- Uses length-aware sampling for sequence length balancing
- Implements MoE with orthogonal regularization

**Data Requirements**:
```csv
Protein_Sequence,CDS_Sequence,Protein_Length
MSTSK...,ATGGTG...,125
```

**Usage**

```
torchrun --nproc_per_node=2 --master_port=49514 actor_train.py
```

**Key Parameters**

```
{
    'd_model': 768,          # Transformer dimension
    'num_experts': 6,        # MoE experts count
    'top_k_experts': 2,      # Active experts per token
    'esm2_dim': 1280,        # ESM-2 feature dimension
    'batch_size': 32,        # Per-GPU batch size
    'entropy_weight': 1.0    # MoE router entropy regularization
}
```
**Output**

- Saved model: `./actor.pt`



### 2. Reward Model Training (`actor_train.py`)

**Function**:  
Trains a quality prediction model for mRNA sequences evaluating:
- Thermodynamic stability (MFE)
- Translation efficiency (CAI)


**Architecture**:
- MoE-enhanced Transformer encoder
- Multi-objective reward combination


**Data Requirements**:
```csv
CDS_Sequence,CDS_Length,CDS_MFE,CDS_CAI
ATGGTG...,300,-125.5,0.87
```

**Usage**

```
torchrun --nproc_per_node=2 --master_port=49516 reward_train.py
```

**Key Parameters**

```
{
    'mfe_weight': 2.0,       # MFE loss coefficient
    'cai_weight': 1.0,       # CAI loss coefficient
    'target_cai': 0.8,       # CAI optimization target
    'ortho_weight': 1000.0   # MoE orthogonal loss weight
}
```
**Output**

- Saved model: `./reward.pt`

### 3. GRPO Training (grpo_train.py)

** Function:**
Performs policy optimization using:

- Group-wise Relative Policy Optimization (GRPO)
- Pretrained actor and reward models
- Stabilized policy updates

** Key Advantages:** 

- ✅ Better sample efficiency
- ✅ Improved training stability
- ✅ Adaptive entropy control

** Configuration: ** 

```
{
    'group_size': 16,        # Experience sharing group size
    'kl_coefficient': 0.01,  # KL divergence penalty
    'policy_update_interval': 50  # Old policy sync frequency
}
```

**Usage**

```
torchrun --nproc_per_node=2 --master_port=49515 grpo_train.py
```

**Output**

- Saved model: `./actor_grpo_epoch{}.pt`


## Workflow Diagram

<div align="center">
  <img src="https://github.com/ShenggengLin/RNARL/raw/main/pictures/workflowDiagram.png" alt="Workflow Diagram" width="70%"/>
</div>

## Configuration

Edit training parameters in each script's `Config` class:
- Batch size
- Learning rate
- Model dimensions
- MoE experts configuration
- Training epochs

## Dataset Preparation

1. Prepare CSV files with columns:
   - Protein_Sequence
   - CDS_Sequence
   - CDS_Length
   - CDS_MFE
   - CDS_CAI
   - Protein_Length

2. Store datasets in `example_data/` directory:
```
example_data/
├── example_train.csv
├── example_val.csv
└── example_test.csv
```

##  Hardware Recommendations 

| Component |  Recommended       |
|-----------|-------------------|
| GPUs      |  2x A100 80GB      |
| CPU       |  32 cores          |
| Memory    |  256GB DDR4        |
| Storage   |  RAID0 NVMe        |



## Web Interface

To enhance accessibility and promote the use of the RNARL framework, a user-friendly web interface was developed and hosted on [Google Colab](https://colab.research.google.com/drive/1Z_t_Lt9CjqA0aygoNcdRzrThF1MOGBRb). This interface is designed to require no environment setup or programming knowledge, enabling users to readily input protein sequences and obtain generated and optimized RNA sequences. The platform supports both the optimization of RNA sequences corresponding to a single protein and the high-throughput optimization for multiple proteins simultaneously through the upload of CSV files. Additionally, users can select different species or RNA types to perform targeted optimization based on their specific needs.




## Citation

If you use this work in your research, please cite:
```bibtex
@article{rnarl2025,
  title={RNARL},
  author={Shenggeng Lin, Co-authors},
  journal={},
  year={2025}
}
```

## License
[MIT License](LICENSE)
```

This README includes:
- Clear installation instructions
- Concise project structure visualization
- Ready-to-use training commands
- Dataset format requirements
- Configuration guidance
- Citation template
- License information

