# VCSNet

Code for **Accurate Individual Tree Counting and Localization via Vegetation-Contrast and Scale-Aware Deep Networks for Urban&ndash;Rural Tree Inventories**.

VCSNet detects individual trees in RGB aerial imagery. It predicts a point for
each detected tree and counts the retained points to estimate the number of trees
in the image. The model combines three components:

- **VCSM** uses RGB-derived vegetation cues to refine foreground confidence.
- **CAAT** uses point annotations to supervise counts over multiple region sizes
  during training.
- **MBRE** enhances the features used for classification and point localization
  through multiple residual branches.

## Repository Map

| File or directory | Purpose |
| --- | --- |
| [`inference.py`](inference.py) | Predict tree counts and locations from one RGB image. |
| [`models/`](models/) | VCSNet architecture and its three modules. |
| [`configs/`](configs/) | Default settings and model variants, including `Base`. |
| [`train.py`](train.py), [`validate.py`](validate.py), [`test.py`](test.py) | Train and evaluate on your own point-annotated dataset. |

## 1. Set Up the Environment

Install a compatible PyTorch and torchvision pair for your CPU or CUDA
environment, then install the remaining dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## 2. Run Inference on a Sample Image

The [Releases](https://github.com/silence2223/VCSNet/releases) page is the download
location for `vcsnet_julu.pth` and `sample_images.zip` when they are published.
Place the checkpoint in the repository root and extract the archive there to
create `images/`.

Run this command from the repository root. No point annotations are needed.

```bash
python inference.py --checkpoint vcsnet_julu.pth --image images/sample_001.png --device cpu --output outputs/sample_001.json
```

## 3. Read the Outputs

The example writes `outputs/sample_001.json`, which contains the predicted count,
the `(x, y)` coordinates of the retained points in image pixels, their confidence
scores, and the foreground threshold. The command also prints the count.

## 4. Optional: Train and Evaluate on Your Own Data

### Data format and training

Place the images, point annotations, and split lists under your dataset directory.
Each row in `train.txt`, `val.txt`, or `test.txt` gives an image path and its
annotation path, both relative to that directory:

```text
images/example.png labels/example.txt
```

An annotation file contains one `x y` coordinate pair per line. An empty file
represents an image without trees. Use separate, nonoverlapping image lists for
training, validation, and testing.

Training from scratch with these settings requires CUDA and the official
ShanghaiTech Part A initialization checkpoint, a separate file from
`vcsnet_julu.pth`:

```bash
python train.py --variant VCSNet --data-root data --official-init weights/SHTechA.pth --output-dir outputs/train
```

The default settings are in [`configs/vcsnet.json`](configs/vcsnet.json). To train
the baseline with all three modules disabled, use `--variant Base` and a new
output directory. In the default training mode, point distance contributes to
matching. The matched-point MSE is computed but does not enter the weighted
training objective.

### Validation and independent testing

```bash
python validate.py --checkpoint outputs/train/checkpoints/best_mae.pth --data-root data --device cuda --output-dir outputs/val
python test.py --checkpoint outputs/train/checkpoints/best_mae.pth --data-root data --val-decision outputs/val/selection.json --device cuda --output-dir outputs/test
```

`validate.py` evaluates `val.txt` and records the checkpoint and evaluation settings
in `selection.json`. `test.py` evaluates `test.txt` with those settings. Both use
new output directories and save metrics and per-image predictions.

## Troubleshooting

- **Image or checkpoint not found:** Run inference from the repository root and
  check that the checkpoint and extracted image match the paths in the command.
- **Output already exists:** Choose a new output filename for inference or a new
  output directory for training and evaluation.

## Data Availability

The sample archive contains images only. The full Julu dataset is not publicly
available due to confidentiality restrictions. Researchers may contact the
authors to request additional data, subject to applicable permissions.

## Attribution and License

The `Base` implementation and point prediction heads are adapted from
[Tencent P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet).
See [LICENSE](LICENSE) for the upstream attribution and academic research terms.
