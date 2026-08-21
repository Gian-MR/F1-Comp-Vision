# F1 Computer Vision

A computer-vision project for classifying Formula 1 cars by team. The notebooks
use COCO bounding-box annotations to crop cars in memory and train a four-class
PyTorch image classifier for:

- Ferrari
- McLaren
- Mercedes
- Red Bull

## Project structure

```text
F1-Comp-Vision/
├── Models/
│   ├── Example_CNN.ipynb          # Transfer-learning baseline
│   ├── My_CNN.ipynb               # Custom CNN experiments
│   └── Visualizer_Annotationer.ipynb
├── artifacts/
│   ├── f1_team_custom_cnn.pt
│   ├── f1_team_efficientnet_b0.pt
│   └── f1_team_resnet18.pt
└── data/                           # Local dataset; ignored by Git
```

## How it works

Each annotated bounding box becomes one classification example. Images are
opened and cropped when a batch is loaded, so the project does not need to
create a second set of cropped image files. The original `train`, `valid`, and
`test` splits are preserved.

The notebooks automatically select Apple Metal (`mps`), CUDA, or CPU depending
on the hardware available.

## Setup

Create and activate a virtual environment, then install the notebook
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter matplotlib pillow scikit-learn torch torchvision tqdm
```

Place the Roboflow dataset in the following layout:

```text
data/raw/roboflow_f1/
├── train/
│   ├── _annotations.coco.json
│   └── ...image files
├── valid/
│   ├── _annotations.coco.json
│   └── ...image files
└── test/
    ├── _annotations.coco.json
    └── ...image files
```

The category names in the COCO files must match the four supported team names.
The `data/` directory is intentionally ignored by Git.

## Usage

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Recommended workflow:

1. Open `Models/Visualizer_Annotationer.ipynb` and inspect the images and
   bounding boxes.
2. Run `Models/Example_CNN.ipynb` for the transfer-learning baseline, or
   `Models/My_CNN.ipynb` for the custom CNN.
3. Set `RUN_TRAINING = True` in the selected training notebook.
4. Run the cells in order to load the data, train, evaluate, and save a
   checkpoint under `artifacts/`.


## Custom CNN results

The best recorded custom-CNN experiment reached **97.04% test accuracy** and a
**96.85% macro F1-score**. That run used a 224 x 224 input, four convolutional
stages, AdamW, a learning rate of 0.001, a batch size of 32, and 10 epochs.

The notebooks report accuracy, macro precision, macro recall, macro F1, and
cross-entropy loss. See `Models/My_CNN.ipynb` for the complete experiment table
and hyperparameter comparisons.