```markdown
# LieHullDynamics
Lie Geometric Learning for Ship Hull Shape Dynamics Representation

This repository contains the experimental implementation and raw results of a Lie-geometric framework for ship hull shape representation, deformation analysis, geometric retrieval and latent dynamic evolution modeling.

The proposed pipeline integrates:
- Lie geometric latent manifold representation
- PCA-based unified shape embedding
- Lie logarithmic deformation mapping
- Koopman-operator latent dynamics prediction
- Geometry-aware shape retrieval
- Compression-fidelity quantitative evaluation
- Comprehensive ablation studies & complexity analysis

## Overview
Traditional ship hull representation mostly relies on raw Euclidean vertex coordinates, handcrafted geometric descriptors or static dimension reduction.
However, hull deformation naturally presents nonlinear manifold variations, sequential evolution characteristics and inherent structural geometric correlations.

This work constructs a geometry-conscious representation based on Lie latent dynamics. We model continuous hull shape evolution as smooth transformations within a compact low-dimensional latent space.

Full computation pipeline:
```
Raw Hull Mesh
    ↓
Point Cloud Sampling
    ↓
Shape PCA Latent Embedding
    ↓
Lie Log Deformation Representation
    ↓
Koopman Dynamic Modeling
    ├───────────────┬───────────────┐
    ↓                               ↓
Shape Retrieval             Multi-step Evolution Prediction
    ↓
Compression Evaluation / Ablation Study / Computational Complexity Analysis
```

## Main Components
### 1. Hull Shape Representation
All hull geometries are uniformly converted into point-cloud sequences as network input.
| Parameter | Value |
| ---- | ---- |
| Sampled points per hull mesh | 1000 |
| Latent embedding dimension | 45 |
| Sequential evolution samples | 1000 |

### 2. Lie Geometric Deformation Modeling
Deformation between adjacent hull configurations:
\[
\Delta z_t = z_{t+1}-z_t
\]
Lie logarithm mapping captures local geometric evolution trajectories on the latent manifold.

### 3. Koopman Dynamic Prediction
We estimate a linear Koopman operator for latent trajectory propagation:
\[
z_{t+1}=Kz_t
\]
The linearized dynamics support multi-step prediction, eigen-spectrum analysis and continuous latent trajectory modeling.

## Dataset Notice
> ⚠️ **ShipD dataset is NOT included in this repository.**
You need to independently download the ShipD ship hull dataset and place the mesh/point cloud data under the designated data folder before running experiments.
Please check the official dataset access channel to obtain raw hull mesh data.

## Repository Structure
LieHullDynamics/
│
├── code/
│   └── lie_hull_dynamics.py      # ✅ MAIN ENTRY / Main execution script
│
├── data/                          # Place downloaded ShipD dataset here
│
├── exp_tables/                    # Organized experimental CSV outputs
│   ├── dataset/
│   │   └── dataset_stats.csv
│   ├── latent/
│   │   ├── latent_analysis.csv
│   │   └── pca_compression.csv
│   ├── retrieval/
│   │   └── retrieval_baseline.csv
│   ├── ablation/
│   │   └── ablation.csv
│   ├── compression/
│   │   └── compression_ratio.csv
│   └── complexity/
│       └── complexity.csv
│
├── figures/                       # Generated visualization figures
│
├── README.md
└── requirements.txt
## Experiments Description
All experimental outputs will be automatically saved into `exp_tables/` after running the main script.

### Dataset Statistics
Basic metadata of ship hull samples, mesh properties and sampling configurations.

### Latent Space Analysis
Evaluate PCA embedding quality, latent dimension selection and reconstruction compression efficiency.

### Version Recovery & Reconstruction
Metrics: Chamfer Distance, Hausdorff Distance, mesh reconstruction error.

### Retrieval Benchmark
Baseline methods for comparison:
- Euclidean distance matching
- Hash embedding
- Mesh Laplacian Spectrum
- PCA static latent representation
- Random Forest descriptor
- Ours (Lie Log + Koopman Dynamics)

Evaluation metrics: Recall@K, NDCG@K, mAP.

### Compression Analysis
Quantify storage overhead, compression ratio and reconstruction error trade-off.

### Ablation Studies
Module-wise ablation verification:
1. PCA latent baseline
2. Mesh Laplacian geometric feature
3. Lie deformation representation
4. Koopman dynamic modeling
5. Full Lie+Koopman integrated model

### Complexity Analysis
Asymptotic computation cost analysis for PCA embedding, Lie mapping, Koopman estimation and retrieval inference.

## Requirements
Python >= 3.9
```
numpy
scipy
pandas
matplotlib
scikit-learn
trimesh
umap-learn
tqdm
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Experiments
> Precondition: Download ShipD dataset and arrange raw mesh data under `data/` folder.

Execute the main script:
```bash
python code/lie_hull_dynamics.py
```
The script will automatically generate:
- Visualization figures saved in `figures/`
- All experimental CSV tables stored under `exp_tables/`
- Retrieval benchmark logs and prediction results

## Citation
If you find this repository helpful for your research, please cite:
```bibtex
@software{LieHullDynamics,
  title={LieHullDynamics: Lie Geometric Learning for Ship Hull Shape Dynamics},
  author={Haijian Shao etc},
  year={2026},
  url={https://github.com/yourname/LieHullDynamics}
}
```
## Limitations of the Lie-based deformation representation
The proposed Lie-algebraic deformation framework has several practical limitations that should be noted.

The Lie Log–Exp recovery relies on truncated second-order BCH approximation. Reconstruction errors gradually accumulate as the magnitude of hull deformation increases, which restricts its application scenarios under extreme large shape modifications.

In pure geometric compression tasks, the latent representation learned via Lie deformation generators does not deliver obvious advantages over standard PCA in terms of reconstruction error at identical embedding dimensions. The primary value of the Lie framework lies in sequential shape composition, editable deformation control and physical interpretability, rather than simply improving compression fidelity.

The current implementation constructs Lie generators from dataset statistics, without embedding prior physical constraints of hydrodynamics or hull structural characteristics. The extracted deformation modes are data-driven and may lack strict physical consistency under out-of-distribution hull designs.

This method targets static shape sequence representation. It cannot naturally predict long-term design evolution trajectories; combining with dynamic operators such as Koopman may bring extra instability and requires careful regularization.

For downstream geometric retrieval tasks, the Lie latent feature brings limited improvement to basic retrieval accuracy compared with PCA latent features. Its core strength is the additional capability to interpolate, combine and edit hull shapes via continuous deformation generators, which cannot be achieved by static PCA embedding.

## License
MIT License


## Acknowledgements
I am responsible for the full implementation of the code and all experimental analysis presented in this repository.
Place downloaded ShipD ship hull mesh dataset in this folder.
Raw dataset is not distributed within this repository.
```
