# GLID-image-segmentation

Maps glacial lakes from satellite imagery using deep-learning semantic segmentation to support environmental monitoring and hazard assessment, with emphasis on Himalayan regions.

## Project snapshot
- Dataset: GLID — 18,367 image/label pairs (512×512 tiles); classes: supraglacial, proglacial, ice-marginal, unclassified.  
- Implementations and experiments: UNet, DeepLabV3, Swin-based models (training and analysis code present under `project/analysis` and `project/codes`).  
- Reports and notebooks: project report, presentation, and notebooks that demonstrate training, evaluation, and post-training analysis.

## Dataset (GLID)
- Zenodo: https://zenodo.org/records/14838695  
- Summary: 18,367 labeled image–mask pairs (512×512). Four lake classes intended for semantic segmentation model development and evaluation.

## Repository contents (selected)
- [project/reports/DL Project Report.pdf](project/reports/DL%20Project%20Report.pdf) — dataset description, methods, and results  
- [project/reports/Segmentation.pptx](project/reports/Segmentation.pptx) — presentation summarizing approach & results  
- [project/analysis/helper.py](project/analysis/helper.py) — data loader (GLIDDataset), loss functions (Dice+BCE), RunManager training loop, metrics (IoU, pixel accuracy)  
- [project/analysis/unet.py](project/analysis/unet.py) — UNet implementation used in experiments  
- [project/analysis/deeplab.py](project/analysis/deeplab.py) — DeepLab-style implementation used in experiments  
- [project/analysis/swin.py](project/analysis/swin.py) — Swin / transformer-based model definitions  
- [project/analysis/Post Training Analysis.ipynb](project/analysis/Post%20Training%20Analysis.ipynb) — long-form analysis with plots and comparisons  
- [project/analysis/Model Comparison.ipynb](project/analysis/Model%20Comparison.ipynb) — compact model comparison notebook  
- [project/codes/Unet-V2.ipynb](project/codes/Unet-V2.ipynb), [project/codes/DeepLabV3-V2.ipynb](project/codes/DeepLabV3-V2.ipynb), [project/codes/SwinUnet-V2.ipynb](project/codes/SwinUnet-V2.ipynb) — training notebooks / experiments  
- Example plots and overlays: `project/analysis/plot-lr-*.png`, `project/analysis/Image-*-*.png`, `project/analysis/Label-*-*.png`

## Quick start — reproduce experiments locally
1. Download the GLID dataset from Zenodo and place it at `project/data/` with this layout:
project/data/ images/ # RGB tiles (matching filenames) labels/ # binary masks / label images (matching filenames)
2. Create a Python environment and install dependencies (create `requirements.txt` if missing):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # expected deps: torch, torchvision, numpy, pandas, matplotlib, pillow, scikit-learn, tqdm
```
3. Train a model using a notebook or script:
 - Open project/codes/Unet-V2.ipynb (or DeepLabV3-V2.ipynb, SwinUnet-V2.ipynb) and run cells to preprocess, instantiate GLIDDataset (defined in project/analysis/helper.py), create dataloaders, and start training.
 - Example minimal flow (from notebooks):
    - Set `MAIN_DIR or TRAIN_DIR` / `VAL_DIR` / `TEST_DIR` to project/data/….
    - Create `train_dataloader` / `val_dataloader` using GLIDDataset.
    - Instantiate model (UNet(), DeepLabV3Plus(num_classes=1), or Swin-based model), loss (DiceBCELoss), optimizer, and call RunManager.train(...).
4. Inspect results and figures with project/analysis/Post Training Analysis.ipynb.

## Data format and conventions

- Filenames for images and labels must match (e.g., images/0001.png ↔ labels/0001.png).
- Labels are expected as binary masks (single-channel) where lake pixels are 1 and background 0. See helper.py for the dataset loader and transforms.
- Training notebooks resize/normalize tiles to 224×224 (see GLIDDataset.imConvert and transform in helper.py).

## Evaluation & metrics

- Primary metrics: Intersection-over-Union (IoU), Dice coefficient, pixel accuracy.
- The code supports IoU computed per-batch and a global IoU across a dataset. For imbalanced classes, report per-class metrics and use cross-validation where possible. Visual overlays (image / true label / predicted mask) are provided in notebooks and project/analysis images.

## Notes & provenance

- The project report documents dataset preparation, hyperparameters (learning rates swept, example: 1e-3 / 1e-4 / 1e-5), and experimental results. Inspect DL Project Report.pdf for details.
Notebooks contain recorded outputs (execution timestamps present in notebook metadata) that show training runs and example progress logs.
Reproducibility checklist

- Add a small sample of data in project/data/sample/ to verify environment quickly.
- Include fixed random seed (notebooks call set_seed(42) in helper.py).
Provide a run_example.sh that runs a short training pass (1–2 epochs) on the sample tile.
Contributing

Dataset (GLID) DOI / Zenodo: https://zenodo.org/records/14838695
Include formal citation and dataset license text when redistributing or publishing. Add a LICENSE file to this repository to clarify reuse terms.

## Contact

Owner: @aaronjacobpj — https://github.com/aaronjacobpj \
## Colaborators: 