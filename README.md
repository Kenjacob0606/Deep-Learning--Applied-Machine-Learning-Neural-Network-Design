# DEEP: Deep Learning — Open Individual Assessment

## Contents

- `deep.ipynb` — Main Jupyter notebook containing code, results, and written
  answers for Questions 1–5 (linear regression models, decision trees/PCA,
  word embeddings, orbital decay regression, and dice product prediction).
- `deep.pdf` — Exported PDF of the notebook after full execution.
- `predict_decay.py` — Standalone script exposing `predict(parameters)` for
  Question 4 (orbital decay time regression).
- `predict_product.py` — Standalone script exposing `predict(images)` for
  Question 5 (dice face-product prediction from images).
- `weights_decay.pth` — Trained network weights for Question 4 (~330 KB).
- `weights_product.pth` — Trained network weights for Question 5 (~9.6 MB).

Question 6 (font character compression) was not attempted.

## Requirements

- Python 3
- PyTorch
- pandas
- Pillow (PIL)
- scikit-learn, gensim, scipy (for Questions 1–3 in the notebook; exact
  pinned versions for the word embeddings section are installed at the top
  of the relevant notebook cell)

## Running the notebook

Open `deep.ipynb` in Google Colab or Jupyter and run all cells from top to
bottom. All data files referenced by the notebook (e.g. the CSV/image
datasets for Questions 1, 2, 4, and 5) are expected to sit alongside the
notebook, as per the assessment brief.

## Using the prediction scripts

### `predict_decay.py` (Question 4)

```python
import torch
from predict_decay import predict

# parameters: B x 6 tensor
# [altitude_km, mass_kg, cross_section_m2, eccentricity, f10.7_index, drag_coeff]
parameters = torch.tensor([[400.0, 50.0, 1.0, 0.01, 150.0, 2.2]])
decay_days = predict(parameters)  # returns B x 1 tensor, decay time in days
```

`weights_decay.pth` must be present in the working directory. Input
normalisation (log transform on mass/cross-section, standardisation) and
output de-normalisation are handled internally by `predict()`.

### `predict_product.py` (Question 5)

```python
import torch
from predict_product import predict

# images: B x 3 x 128 x 128 tensor, RGB values in (0, 1)
products = predict(images)  # returns B x 1 tensor of predicted products
```

`weights_product.pth` must be present in the working directory.

**Note:** `predict_product.py` reads `dice_images/train/labels.csv` at
import time to rebuild the product-to-class-index mapping used during
training. This file (from the original `dice_images.zip` dataset) must be
present at that relative path for the script to import successfully.

## Architecture summary

- **Question 4 (orbital decay):** Fully-connected feedforward network
  (6 → 64 → 128 → 256 → 128 → 64 → 1) with GELU activations, trained on
  standardised inputs (with log1p transforms on mass and cross-sectional
  area) and a standardised log-target.
- **Question 5 (dice product):** Convolutional neural network (4 conv
  blocks with batch normalisation, ReLU, and max-pooling, 3→32→64→128→256
  channels) followed by fully-connected classification head, trained
  end-to-end from raw pixel input to predicted product class.

Full design justification and discussion for all questions is provided in
`deep.ipynb` / `deep.pdf`.
