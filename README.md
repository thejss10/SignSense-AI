# SignSense AI

A camera-based hand gesture recognition project built with MediaPipe, OpenCV, scikit-learn, and Streamlit. The app detects hand landmarks and uses a trained Random Forest classifier to predict a gesture.

This repository is a practical project for anyone interested in learning about machine learning, computer vision, and Python through a complete data collection, training, and prediction workflow.

## Features

- Detects hand landmarks with MediaPipe
- Converts landmarks into normalized numerical features
- Trains a Random Forest classifier with scikit-learn
- Collects gesture samples from a webcam
- Displays predictions, class probabilities, and landmark visualization in a Streamlit app
- Lets you define and train additional gesture classes using your own examples

## Current gesture classes

The included dataset contains five classes:

- FIST
- OPEN_PALM
- THUMBS_UP
- PEACE
- ONE_FINGER

These are static hand poses. The project does not currently recognize motion-based gestures or translate sign language.

## How it works

1. **Landmark detection:** MediaPipe Hands detects 21 landmarks on a hand.
2. **Feature extraction:** Each landmark contributes its x, y, and z coordinates, producing 63 numerical features.
3. **Normalization:** The wrist landmark is used as the origin, and coordinates are scaled by hand size.
4. **Model training:** A Random Forest classifier learns to map the landmark features to gesture labels.
5. **Prediction:** The Streamlit app processes a captured camera image and displays the predicted class and the model's class probabilities.

## Learning topics

By exploring the code, you can learn about:

- Computer vision and hand landmark detection
- Turning visual information into machine learning features
- Feature scaling and normalization
- Labeled datasets and data collection
- Supervised learning and multi-class classification
- Random Forests, an ensemble of decision trees
- Training and test sets, stratified splitting, and reproducibility
- Class balance and the `class_weight="balanced"` option
- Model evaluation with accuracy, precision, recall, F1 score, and a confusion matrix
- Prediction probabilities and how to present model output in an application
- Building a user interface with Streamlit

## Project structure

```text
SignSense-AI/
├── app.py               # Streamlit prediction app
├── camera_test.py       # Checks webcam access with OpenCV
├── collect_data.py      # Collects labeled hand landmark samples
├── train_model.py       # Trains and evaluates the Random Forest model
├── gestures.csv         # Labeled feature data used for training
└── gesture_model.pkl    # Trained model and feature names used by the app
```

## Requirements

- Python 3.10 or another version compatible with the project dependencies
- A webcam for collecting data and using the camera features
- Python packages listed in `requirements.txt`

## Setup

Download or clone the repository, then open a terminal in the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project directory, start Streamlit:

```bash
streamlit run app.py
```

Open the local address printed in the terminal, allow camera access, and capture a hand pose. The app displays the predicted gesture, model confidence, probability chart, and detected landmarks.

## Collect your own gesture data

You can add gesture classes and train the model on examples you collect:

1. Open `collect_data.py` and edit the `GESTURES` list to contain the labels you want to recognize.
2. Adjust `SAMPLES_PER_GESTURE` if you want to collect a different number of examples for each class.
3. From the project directory, run:

   ```bash
   python collect_data.py
   ```

4. For each class, show the pose to the webcam and press **S** in the camera window to start collecting. Press **Q** to stop.
5. Run the training script:

   ```bash
   python train_model.py
   ```

6. Start the app again with `streamlit run app.py` to use the newly trained model.

**Important:** `collect_data.py` opens `gestures.csv` in write mode, so running it replaces the existing dataset. Back up the CSV first if you want to keep it. Collect examples for every class you plan to recognize, and vary the hand position, distance, and lighting where possible. The dataset needs at least two gesture classes for training.

## Model evaluation

`train_model.py` splits the available data into training and test sets, trains the classifier, and prints accuracy, a classification report, and a confusion matrix. The model's reported performance depends on the dataset and split; it does not guarantee the same accuracy on new people, cameras, backgrounds, or lighting conditions.

## Limitations

- The current model recognizes static hand poses, not full sign language.
- Only one hand is processed at a time.
- Recognition quality depends on the examples in the training data and the camera conditions.
- The model's probability output is a classifier score and may not be a calibrated measure of real-world certainty.

## Future ideas

- Add more gesture classes and collect a broader, more varied dataset
- Add a workflow that appends data instead of replacing the existing CSV
- Evaluate with samples collected from people and environments not represented in training
- Add automated tests and more detailed model analysis
- Explore recognition of motion-based gestures

## License

No license has been selected for this repository yet. Add a `LICENSE` file to specify how others may use, modify, or distribute the project.
