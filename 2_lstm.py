# --- 1. CELINA: IMPORTI ---
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Input, Dropout, 
                                     BatchNormalization, Conv1D, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# --- 2. CELINA: PODEŠAVANJA ---
RECORDINGS_PATH = Path('recordings')
N_MFCC = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 3. CELINA: FUNKCIJA ZA OBRADU PODATAKA ---
def load_and_process_data(path, n_mfcc, test_size, random_state):
    X_data = []
    y_labels = []
    
    print("Počinjem učitavanje i obradu fajlova...")
    
    for i, wav_file in enumerate(path.glob('*.wav')):
        try:
            filename = wav_file.stem
            label = filename.split('_')[0]
            
            # Učitaj audio
            data, samplerate = librosa.load(wav_file, sr=22050)  # Fiksiran sample rate
            
            # Izračunaj MFCC i transponuj
            mfccs = librosa.feature.mfcc(y=data, sr=samplerate, n_mfcc=n_mfcc)
            mfccs_transposed = np.transpose(mfccs)
            
            X_data.append(mfccs_transposed)
            y_labels.append(label)
            
            if (i + 1) % 500 == 0:
                print(f"Obrađeno {i + 1} fajlova...")
                
        except Exception as e:
            print(f"Greška pri obradi fajla {wav_file}: {e}")
            
    print(f"Obrada završena. Učitano {len(X_data)} fajlova.")

    # --- 3.1. Padding ---
    max_len = max(mfcc.shape[0] for mfcc in X_data)
    print(f"Maksimalna dužina sekvence: {max_len}")
    X_padded = pad_sequences(X_data, maxlen=max_len, padding='post', dtype='float32')
    
    # --- 3.2. Kodiranje Labela ---
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)
    num_classes = len(encoder.classes_)
    print(f"Pronađeno {num_classes} jedinstvenih klasa.")

    # --- 3.3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, 
        y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded
    )

    # --- 3.4. NORMALIZACIJA ---
    print("\nNormalizujem podatke...")
    train_shape = X_train.shape
    test_shape = X_test.shape

    X_train_2d = X_train.reshape(-1, train_shape[2])
    X_test_2d = X_test.reshape(-1, test_shape[2])
    
    scaler = StandardScaler()
    scaler.fit(X_train_2d)
    
    X_train_scaled_2d = scaler.transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)
    
    X_train = X_train_scaled_2d.reshape(train_shape)
    X_test = X_test_scaled_2d.reshape(test_shape)
    
    print("Normalizacija završena.")
    
    # PROVERA PODATAKA
    print("\n--- PROVERA PODATAKA ---")
    print(f"X_train min: {X_train.min():.3f}, max: {X_train.max():.3f}")
    print(f"X_train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
    print(f"Distribucija klasa: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test, max_len, num_classes, encoder

# --- 4. CELINA: GLAVNI PROGRAM ---
if __name__ == "__main__":
    
    # Pokreni obradu podataka
    X_train, X_test, y_train, y_test, MAX_LEN, NUM_CLASSES, encoder = load_and_process_data(
        path=RECORDINGS_PATH, 
        n_mfcc=N_MFCC,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    print("\n--- KONAČNI OBLICI PODATAKA ---")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    
    # --- 5. CELINA: MODEL SA BATCH NORMALIZATION ---
    
    print("\n=== OPCIJA 1: LSTM sa Batch Normalization ===")
    model1 = Sequential([
        Input(shape=(MAX_LEN, N_MFCC)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print("\n=== OPCIJA 2: Conv1D + LSTM (Preporučeno) ===")
    model2 = Sequential([
        Input(shape=(MAX_LEN, N_MFCC)),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Izaberi model (probaj oba!)
    model = model2  # Promeni u model1 ako želiš drugi
    
    model.summary()

    # --- 6. CELINA: KOMPAJLIRANJE ---
    print("\nKompajliranje modela...")
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # SMANJEN learning rate
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # --- 7. CELINA: CALLBACKS ---
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # --- 8. CELINA: TRENIRANJE ---
    print("\nPočinjem treniranje...")
    
    EPOCHS = 100  # Povećano, ali će early stopping zaustaviti ako ne napreduje
    BATCH_SIZE = 32

    history = model.fit(
        X_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )

    print("\nTrening završen!")

    # --- 9. CELINA: EVALUACIJA ---
    print("\nEvaluacija modela...")
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n{'='*50}")
    print(f"FINALNA TAČNOST NA TEST PODACIMA: {test_acc * 100:.2f}%")
    print(f"{'='*50}")

    # --- 10. CELINA: VIZUALIZACIJA ---
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Trening', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validacija', linewidth=2)
    plt.title('Tačnost Modela', fontsize=16, fontweight='bold')
    plt.xlabel('Epoha', fontsize=12)
    plt.ylabel('Tačnost', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trening', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validacija', linewidth=2)
    plt.title('Gubitak Modela', fontsize=16, fontweight='bold')
    plt.xlabel('Epoha', fontsize=12)
    plt.ylabel('Gubitak', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 11. SAČUVAJ MODEL ---
    model.save('fsdd_model.keras')
    print("\nModel sačuvan kao 'fsdd_model.keras'")
    
    # --- 12. TEST PREDIKCIJA ---
    print("\n--- TEST PREDIKCIJA NA NEKOLIKO PRIMERA ---")
    predictions = model.predict(X_test[:5])
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("Stvarne klase:", y_test[:5])
    print("Predviđene:   ", predicted_classes)
    print("Tačno?" , predicted_classes == y_test[:5])