# DeepCave: Cave Entrance Identification System

This project implements a machine learning pipeline for identifying caves from
entrance photos. It uses **DINOv2 embeddings** with a **custom-trained linear
classification head**. The pipeline includes:

- Data preparation and training
- Inference and cave identification
- Model export for mobile/offline use
- A GUI tool for manual photo classification and labeling

## Repository Structure

| File | Purpose |
|------|---------|
| training_linear_layer.py | Trains a linear classification head on top of
DINOv2 features with error handling for corrupted images. |
| final_inference.py | Runs inference on cave images, fetching metadata from
Airtable and producing predictions. |
| export_models.py | Exports the trained model and classification head to a
format usable on mobile devices (TorchScript). |
| photo_class_with_cut_and_cave_name.py | Tkinter-based GUI tool for manually
categorizing images into predefined categories. |

## Installation

bash:
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt


**Dependencies include:**
- torch
- torchvision
- transformers
- Pillow
- tqdm
- requests
- urllib3

## Training the Model

bash:
python training_linear_layer.py

This will:
1. Load DINOv2 as a feature extractor.
2. Train a linear classification layer.
3. Save the trained weights and cave index map.

Bad images are logged to:
- bad_images.json
- bad_images.txt

## Running Inference

bash:
python final_inference.py

This script:
- Loads the trained model.
- Processes input images.
- Queries Airtable for cave metadata.
- Outputs predictions with confidence scores.

## Exporting the Model for Mobile

bash:
python export_models.py

This:
- Loads the trained model and linear head.
- Optimizes for mobile deployment.
- Saves TorchScript model files.

## Using the GUI Tool

bash
python photo_class_with_cut_and_cave_name.py


The GUI allows:
- Categorizing images into predefined cave-related classes.
- Updating the database with new classifications.

## Known Limitations

- Requires GPU for efficient training/inference.
- Airtable API credentials are hardcoded — must be replaced with your own.
- The GUI currently uses a fixed set of categories.

## License

MIT License (or specify your license).
