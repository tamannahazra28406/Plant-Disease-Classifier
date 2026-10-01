# 🌿 Plant Disease Classifier

Fine-tune a pretrained CNN (EfficientNet / ResNet50) on leaf images to detect 38 crop diseases, and serve it as a Gradio demo where anyone can upload a leaf photo.

**Pipeline:** Dataset → Preprocess/Augment → Fine-tune → Evaluate → Deploy → Use

## 1. Setup
```bash
pip install -r requirements.txt
```
On Google Colab: Runtime → Change runtime type → T4 GPU, then `!pip install -q timm gradio`.

## 2. Dataset (PlantVillage, 38 classes)
Download **"New Plant Diseases Dataset (Augmented)"** from Kaggle
(`vipoooool/new-plant-diseases-dataset`). It contains `train/` and `valid/` folders.

```bash
pip install kaggle   # put kaggle.json in ~/.kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data --unzip
```
Then find the folder containing `train/` and `valid/` (usually `data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)`).

Any folder of `class_name/*.jpg` also works (it is auto-split 80/20). Use fewer classes (15+) by deleting class folders if you want a smaller demo.

## 3. Train
```bash
python train.py --data_dir "<path-to-dataset>" --model efficientnet_b0
# ResNet50 alternative:
python train.py --data_dir "<path-to-dataset>" --model resnet50 --batch_size 32
```
Two phases: (1) freeze the backbone and train only the new head, (2) unfreeze and fine-tune at a 10x lower LR. Expect ~95%+ validation accuracy in ~10 epochs on a free Colab T4.

## 4. Evaluate
```bash
python evaluate.py --data_dir "<path-to-dataset>"
```
Prints precision/recall per class and saves `confusion_matrix.png`. Use it to compare ResNet50 vs EfficientNet.

## 5. Deploy the demo
```bash
python app.py            # local: http://127.0.0.1:7860
python app.py --share    # public link (Colab)
```
Free hosting: create a Hugging Face Space (Gradio SDK), upload `app.py`, `common.py`, `requirements.txt` and `checkpoints/best.pt`.

## Concepts this project teaches
- **Transfer learning:** ImageNet features already know edges, textures, and spots.
- **Freezing layers:** train the small head first so noisy gradients don't wreck pretrained weights.
- **Augmentation:** flips, rotations, crops, and color jitter make the model robust to real phone photos.
- **Label smoothing + AdamW + cosine LR + mixed precision:** cheap wins for stable training.

## Caveat
PlantVillage images are lab-style (single leaf, plain background), so accuracy on messy field photos will be lower. To make it stronger, add field images (e.g. PlantDoc) or Indian crop data and retrain.

## Ideas to extend
- Grad-CAM heatmaps to show *where* the model looks
- Treatment recommendations per disease
- Export to ONNX/TFLite for a mobile app
