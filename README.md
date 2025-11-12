# Comparative analysis of deep neural networks for speech recognition

This project dives into the performances of multiple neural network architectures used for audio classifications (Key spotting) on FSDD (Free Spoken Digit Dataset).

The goal was to compare not only basic machine learning methods, but also to analyze the trade-offs of modern Deep Learning architectures, including CRNN, Efficient CNN and Self-Attention based models.

## Results:

The Deep Learning models outperform traditional machine learning algorithms by a mile, all while attaining almost perfect accuracy.

| Model | Architecture type | Numer of parameters | Accuracy (Test Set) |
| :--- | :--- | :--- | :--- |
| **Classical methods** |
| Logistic regression | Classical ML | - | 66.33% |
| SVM | Classical ML | - | 70.11% |
| Random Forest | Classical ML | - | 82.67% |
| XGBoost | Classical ML | - | 85.90% |
| **Deep Learning models** |
| Regular CNN model | CNN | - | 97.28% |
| **Newer models** |
| Efficient CNN | `SeparableConv1D` | **TO BE ADDED** | **TO BE ADDED** |
| CRNN| `CNN + LSTM` | **~185,082** | **99.33%** |
| Transformer | `LSTM + Attention` | **TO BE ADDED** | **TO BE ADDED** |

# Visual Training Results #
All three custom-built models demonstrate successful training and excellent generalization. The training (blue) and validation (orange) curves are nearly identical, indicating that aggressive regularization (BatchNormalization and Dropout) effectively prevented overfitting.

1. CRNN (CNN-LSTM) Model (Baseline)
This model established our baseline performance, achieving 99.33% accuracy.

2. Efficient CNN (SeparableConv1D) Model
This model was designed to test efficiency. It learns significantly faster (reaching 90% accuracy by epoch 15) and has a fraction of the parameters.

3. Attention (Transformer) Model
This model uses a modern MultiHeadAttention layer to "focus" on the most important parts of the audio sequence, also achieving near-perfect accuracy.

# Project structure
To maintain clean and reusable code, the project is structured as follows:

- data_processor.py: A single, shared module containing the

- load_and_process_data function. This function handles all audio loading (Librosa), MFCC extraction, padding, StandardScaler normalization, and splitting data into train/test sets.

- 2_lstm.py: Script to train and evaluate the baseline CRNN (CNN-LSTM) model.

- 3_efikasni_cnn.py: Script to train and evaluate the Efficient CNN model.

- 4_self_attention.py: Script to train and evaluate the Attention-based model.

- requirements.txt: All necessary Python dependencies.

- results/: Directory where training graphs are saved.

# How to run
Clone the repository:

```
git clone https://github.com/dimitrijepesic/speech-recognition-2.0.git
cd speech-recognition-2.0
```
Create a virtual environment and install dependencies:


```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
Download the data: Download the Free Spoken Digit Dataset (FSDD) and place all .wav files into a folder named recordings/ in the root of the project.

Run a model training script:

```
# To train the CRNN (CNN-LSTM) model
python 2_lstm.py

# To train the Efficient CNN model
python 3_efikasni_cnn.py

# To train the Attention model
python 4_self_attention.py
```

Each script will automatically process the data, train the model, save the final .keras file to saved_models/, and save the training graph to results/.


## Tech Stack

**Core language:** *Python 3.11*

**Deep learning:** *TensorFlow / Keras* For building and training all neural network architectures

**Audio and data processing:** *Librosa, Scikit-learn, NumPy* Used for MFCC feature extraction, data normalization (StandardScaler), numerical operations, and data splitting

**Visualization:** *Matplotlib* Used for visualizing training history (accuracy and loss curves)