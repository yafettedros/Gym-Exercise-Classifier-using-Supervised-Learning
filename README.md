# Gym-Exercise-Classifier-using-Supervised-Learning

This project applies supervised learning to classify different types of gym exercises based on video data. Using the Gym Workout/Exercises Video Dataset from Kaggle, I extracted frames from exercise videos and trained a Convolutional Neural Network (CNN) to identify which exercise was being performed.

## Dataset
- **Source**: [Kaggle - Gym Workout/Exercises Video Dataset](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video)
- **Classes**: Push-Up, Squat, Deadlift, Pull-Up, Bench Press
- **Data Format**: Videos in .mp4 format grouped into folders by exercise type (used as labels).

## Tools and Libraries
- Python
- OpenCV
- TensorFlow 
- Pandas, NumPy
- Matplotlib

## Directory Structure
```
project_folder/
├── gym_videos/              # Extracted video dataset folders
├── frames/                  # Extracted frames from videos
├── model/                   # Saved model files
├── train_model.py           # Model training script
├── extract_frames.py        # Frame extraction script
├── evaluate_model.py        # Misclassification evaluator
└── README.md
```

## Step-by-Step Instructions

### 1. Download and Extract Dataset
```bash
kaggle datasets download -d philosopher0808/gym-workoutexercises-video
unzip gym-workoutexercises-video.zip -d gym_videos
```

### 2. Run Frame Extraction
See `extract_frames.py` for logic used to convert videos to image frames.

### 3. Train the Model
Use `train_model.py` to train a CNN using image data from `frames/`.

### 4. Evaluate Results
Run `evaluate_model.py` to review accuracy and identify misclassified samples.

## Results Summary
| Exercise     | Precision | Recall | F1 Score | Samples |
|--------------|-----------|--------|----------|---------|
| Push-Up      | 0.89      | 0.91   | 0.90     | 50      |
| Squat        | 0.87      | 0.84   | 0.85     | 50      |
| Deadlift     | 0.82      | 0.79   | 0.80     | 40      |
| Pull-Up      | 0.85      | 0.83   | 0.84     | 45      |
| Bench Press  | 0.80      | 0.78   | 0.79     | 45      |

## Limitations
- Only used image frames, not full video sequences.
- Visual overlap in some exercises caused misclassifications.
- Class imbalance required manual checks.
