# StyleTransfer

A simple neural style transfer project built with PyTorch and Torchvision. It uses a pretrained VGG19 feature extractor to combine the content of one image with the visual style of another image.

## Project Structure

```text
.
+-- main.py                 # Main style transfer script
+-- modelProcessing.py      # VGG19 feature extractor and loss functions
+-- extraProcessing.py      # Image loading and saving helpers
+-- test.py                 # Utility script for resizing an output image
+-- requirement.txt         # Python dependencies
`-- image/                  # Input images
    +-- content.png
    +-- content2.jpg
    `-- style.jpg
```

Generated images are written to `output1/` by default. Intermediate snapshots are saved as `output1/round_<step>.jpg`.

## Requirements

- Python 3.10 or newer is recommended
- PyTorch
- Torchvision
- Pillow
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirement.txt
```

If you want GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide.

## Usage

1. Put your content image and style image in the `image/` folder.
2. Update the paths and parameters at the top of `main.py` if needed:

```python
CONTENT_PATH = 'image/content.png'
STYLE_PATH = 'image/style.jpg'
OUTPUT_PATH = 'output1/output.jpg'
```

3. Run the transfer:

```bash
python main.py
```

The script will download pretrained VGG19 weights automatically the first time it runs, then save progress images and the final result under `output1/`.

## Main Parameters

- `IMG_SIZE`: working image size. Larger values produce more detail but require more memory and time.
- `STEPS`: number of optimization steps.
- `STYLE_WEIGHT`: style loss weight. Higher values make the output closer to the style image.
- `LR`: Adam optimizer learning rate.
- `SHOW_EVERY`: interval for saving intermediate result images.

## Notes

- The current implementation runs on CPU unless the code is extended to move tensors and the model to CUDA.
- `output/`, `output1/`, local virtual environments, caches, and personal notes are ignored by Git.
- Input images in `image/` are kept trackable so the repository can include small sample images. Remove or replace them before publishing if they are private or copyrighted.
