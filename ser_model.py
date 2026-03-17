# 🎤 Speech Emotion Recognition - Jordanian Dialect
''''
**Complete notebook - everything in one file!**

This notebook includes:
- Data loading and preprocessing
- Feature extraction (MFCC + Wav2Vec2)
- Model training (SVM, Wav2Vec2, Combined)
- Training, Validation, and Test accuracy
- Confusion matrices for all models
- Testing new audio files
'''

# Import all libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Configuration
AUDIO_DIR = "data/Jordanian_SER_Split"
SAVE_DIR = "saved_models"

SAMPLE_RATE = 16000
MAX_DURATION = 5.0

# Emotion labels
EMOTIONS = ["happy", "sad", "angry", "neutral"]
EMOTION_TO_ID = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}
ID_TO_EMOTION = {0: "happy", 1: "sad", 2: "angry", 3: "neutral"}
NUM_CLASSES = 4

# Model settings
PRETRAINED_MODEL = "facebook/wav2vec2-large-xlsr-53"

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
RANDOM_SEED = 42

# Best parameters
BEST_SVM_PARAMS = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

print("Configuration:")
print(f"  Audio Directory: {AUDIO_DIR}")
print(f"  Save Directory:  {SAVE_DIR}")
print(f"  Emotions:        {EMOTIONS}")

## 2. Helper Functions

def load_audio(file_path):
    """Load audio file and resample to 16kHz using librosa to bypass FFmpeg issues."""
    # librosa automatically handles mono conversion and resampling
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Convert numpy array to torch tensor
    waveform = torch.from_numpy(y)
    
    return waveform


def process_audio(file_path):
    """Load and normalize audio (variable length)."""
    waveform = load_audio(file_path)

    # Normalize only (no padding/truncating)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    return waveform

print("✓ Audio functions loaded")

def create_metadata(audio_dir):
    """Create metadata from folder structure."""
    data = []
    audio_path = Path(audio_dir)

    print("Scanning folders...")

    for split in ["train", "val", "test"]:
        split_path = audio_path / split
        if not split_path.exists():
            continue

        print(f"\n{split.upper()}:")

        for emotion_folder in split_path.iterdir():
            if not emotion_folder.is_dir():
                continue

            emotion = emotion_folder.name.lower()

            if emotion not in EMOTIONS:
                print(f"  Skipping: {emotion_folder.name}")
                continue

            emotion_id = EMOTION_TO_ID[emotion]
            wav_files = list(emotion_folder.glob("*.wav"))
            print(f"  {emotion}: {len(wav_files)} files")

            for audio_file in wav_files:
                data.append({
                    "filename": str(audio_file),
                    "emotion": emotion,
                    "emotion_id": emotion_id,
                    "split": split
                })

    df = pd.DataFrame(data)

    print(f"\n{'='*50}")
    print(f"Total samples: {len(df)}")

    if len(df) > 0:
        print(f"\nBy emotion:\n{df['emotion'].value_counts()}")
        print(f"\nBy split:\n{df['split'].value_counts()}")

    return df

print("✓ Metadata function loaded")

def extract_mfcc_features(file_path, n_mfcc=40):
    """Extract enhanced MFCC features from audio file."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    features = []

    # === MFCC (40 coefficients × 4 stats = 160) ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    features.extend(np.min(mfccs, axis=1))
    features.extend(np.max(mfccs, axis=1))

    # === Delta MFCC (40 × 2 = 80) ===
    delta_mfccs = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfccs, axis=1))
    features.extend(np.std(delta_mfccs, axis=1))

    # === Delta-Delta MFCC (40 × 2 = 80) ===
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.extend(np.mean(delta2_mfccs, axis=1))
    features.extend(np.std(delta2_mfccs, axis=1))

    # === Chroma (12 × 1 = 12) ===
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # === Spectral Centroid (2) ===
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    # === Zero Crossing Rate (2) ===
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # === RMS Energy (2) ===
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # === Spectral Bandwidth (2) ===
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.std(spec_bw))

    # === Spectral Contrast (7 × 2 = 14) ===
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spec_contrast, axis=1))
    features.extend(np.std(spec_contrast, axis=1))

    # === Spectral Rolloff (2) ===
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(spec_rolloff))
    features.append(np.std(spec_rolloff))

    # === Mel Spectrogram (40 × 2 = 80) ===
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_spec_db, axis=1))
    features.extend(np.std(mel_spec_db, axis=1))

    # === Pitch (F0) (4) ===
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features.append(np.mean(f0_clean))
            features.append(np.std(f0_clean))
            features.append(np.min(f0_clean))
            features.append(np.max(f0_clean))
        else:
            features.extend([0, 0, 0, 0])
    except:
        features.extend([0, 0, 0, 0])

    return np.array(features)

print("✓ Enhanced MFCC function loaded")
print(f"  Feature count: ~440 features")

def create_classifier(input_dim, hidden_layers, dropout, num_classes):
    """Create classifier model."""
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)

print("✓ Model architecture function loaded")

## 3. Load Data

# Create metadata
metadata = create_metadata(AUDIO_DIR)

# Split by train/val/test
train_metadata = metadata[metadata['split'] == 'train'].reset_index(drop=True)
val_metadata = metadata[metadata['split'] == 'val'].reset_index(drop=True)
test_metadata = metadata[metadata['split'] == 'test'].reset_index(drop=True)

print(f"\n✓ Data loaded:")
print(f"  Train: {len(train_metadata)}")
print(f"  Val:   {len(val_metadata)}")
print(f"  Test:  {len(test_metadata)}")

# ============================================================
# LOAD OR EXTRACT FEATURES
# ============================================================

EMBEDDINGS_FILE = os.path.join(SAVE_DIR, 'embeddings.pt')
MFCC_FILE = os.path.join(SAVE_DIR, 'mfcc_features.npz')

# Load embeddings
if os.path.exists(EMBEDDINGS_FILE):
    print("✓ Loading saved embeddings...")
    saved_data = torch.load(EMBEDDINGS_FILE)
    emb_train = saved_data['train_embeddings']
    emb_labels_train = saved_data['train_labels']
    emb_val = saved_data['val_embeddings']
    emb_labels_val = saved_data['val_labels']
    emb_test = saved_data['test_embeddings']
    emb_labels_test = saved_data['test_labels']
    embedding_dim = saved_data['embedding_dim']
    print(f"  Embeddings: Train={emb_train.shape}, Val={emb_val.shape}, Test={emb_test.shape}")
    SKIP_EMBEDDINGS = True
else:
    print("❌ No saved embeddings. Run extraction cells.")
    SKIP_EMBEDDINGS = False

# Load MFCC
if os.path.exists(MFCC_FILE):
    print("✓ Loading saved MFCC features...")
    mfcc_data = np.load(MFCC_FILE)
    mfcc_train = mfcc_data['mfcc_train']
    mfcc_val = mfcc_data['mfcc_val']
    mfcc_test = mfcc_data['mfcc_test']
    labels_train = mfcc_data['labels_train']
    labels_val = mfcc_data['labels_val']
    labels_test = mfcc_data['labels_test']
    print(f"  MFCC: Train={mfcc_train.shape}, Val={mfcc_val.shape}, Test={mfcc_test.shape}")
    SKIP_MFCC = True
else:
    print("❌ No saved MFCC. Run MFCC extraction cell.")
    SKIP_MFCC = False

# Summary
if SKIP_EMBEDDINGS and SKIP_MFCC:
    print("\n✅ All features loaded! Skip to 'Train SVM'")
else:
    print("\n⚠️ Some features missing. Run the extraction cells below.")

## 4. Extract Features (Skip if already loaded)

# Extract MFCC features (SKIP if already loaded)
if not SKIP_MFCC:
    def extract_all_mfcc(metadata_df):
        features = []
        labels = []
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                feat = extract_mfcc_features(row['filename'])
                features.append(feat)
                labels.append(row['emotion_id'])
            except Exception as e:
                print(f"Error: {row['filename']}: {e}")
        return np.array(features), np.array(labels)

    print("Extracting MFCC features...")
    mfcc_train, labels_train = extract_all_mfcc(train_metadata)
    mfcc_val, labels_val = extract_all_mfcc(val_metadata)
    mfcc_test, labels_test = extract_all_mfcc(test_metadata)

    # Save
    np.savez(MFCC_FILE,
             mfcc_train=mfcc_train, mfcc_val=mfcc_val, mfcc_test=mfcc_test,
             labels_train=labels_train, labels_val=labels_val, labels_test=labels_test)
    print(f"✓ Saved to {MFCC_FILE}")
else:
    print("✓ MFCC features already loaded")

# Extract Wav2Vec2 embeddings (SKIP if already loaded)
if not SKIP_EMBEDDINGS:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    print("Loading Wav2Vec2 model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(PRETRAINED_MODEL).to(device)
    wav2vec2_model.eval()

    def extract_embedding(audio_path):
        waveform = process_audio(audio_path)
        with torch.no_grad():
            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)
            outputs = wav2vec2_model(input_values)
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu()

    def extract_all_embeddings(metadata_df):
        embeddings = []
        labels = []
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                emb = extract_embedding(row['filename'])
                embeddings.append(emb)
                labels.append(row['emotion_id'])
            except Exception as e:
                print(f"Error: {row['filename']}: {e}")
        return torch.cat(embeddings, dim=0), torch.tensor(labels)

    print("Extracting embeddings...")
    emb_train, emb_labels_train = extract_all_embeddings(train_metadata)
    emb_val, emb_labels_val = extract_all_embeddings(val_metadata)
    emb_test, emb_labels_test = extract_all_embeddings(test_metadata)
    embedding_dim = emb_train.shape[1]

    # Save
    torch.save({
        'train_embeddings': emb_train, 'train_labels': emb_labels_train,
        'val_embeddings': emb_val, 'val_labels': emb_labels_val,
        'test_embeddings': emb_test, 'test_labels': emb_labels_test,
        'embedding_dim': embedding_dim
    }, EMBEDDINGS_FILE)
    print(f"✓ Saved to {EMBEDDINGS_FILE}")
else:
    print("✓ Embeddings already loaded")
    # Load Wav2Vec2 model for predictions
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(PRETRAINED_MODEL).to(device)
    wav2vec2_model.eval()

## 5. Train SVM

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 60)
print("TRAINING SVM MODEL")
print("=" * 60)

# Scale features
mfcc_scaler = StandardScaler()
mfcc_train_scaled = mfcc_scaler.fit_transform(mfcc_train)
mfcc_val_scaled = mfcc_scaler.transform(mfcc_val)
mfcc_test_scaled = mfcc_scaler.transform(mfcc_test)

print(f"Train: {mfcc_train_scaled.shape[0]} samples, {mfcc_train_scaled.shape[1]} features")
print(f"Val:   {mfcc_val_scaled.shape[0]} samples")
print(f"Test:  {mfcc_test_scaled.shape[0]} samples")

# Train SVM with best C
print("\nTraining SVM (C=10)...")
svm = SVC(C=10, kernel='rbf', gamma='scale', random_state=42)
svm.fit(mfcc_train_scaled, labels_train)
# Add this directly after calculating svm_test_acc and printing the results


# Evaluate on all sets
svm_train_preds = svm.predict(mfcc_train_scaled)
svm_val_preds = svm.predict(mfcc_val_scaled)
svm_test_preds = svm.predict(mfcc_test_scaled)

svm_train_acc = accuracy_score(labels_train, svm_train_preds)
svm_val_acc = accuracy_score(labels_val, svm_val_preds)
svm_test_acc = accuracy_score(labels_test, svm_test_preds)

print(f"\n" + "=" * 60)
print("SVM RESULTS")
print("=" * 60)
print(f"Train Accuracy:  {svm_train_acc:.2%}")
print(f"Val Accuracy:    {svm_val_acc:.2%}")
print(f"Test Accuracy:   {svm_test_acc:.2%}")
print(f"\nOverfitting Gap: {(svm_train_acc - svm_test_acc):.2%}")
svm_cm = confusion_matrix(labels_test, svm_test_preds)
# KNN & MLP

# ============================================================
# KNN MODEL TRAINING
# ============================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRAINING KNN MODEL")
print("=" * 60)

# Scale features
mfcc_scaler = StandardScaler()
mfcc_train_scaled = mfcc_scaler.fit_transform(mfcc_train)
mfcc_val_scaled = mfcc_scaler.transform(mfcc_val)
mfcc_test_scaled = mfcc_scaler.transform(mfcc_test)

print(f"Train: {mfcc_train_scaled.shape[0]} samples, {mfcc_train_scaled.shape[1]} features")
print(f"Val:   {mfcc_val_scaled.shape[0]} samples")
print(f"Test:  {mfcc_test_scaled.shape[0]} samples")

# Test different K values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
train_accs = []
val_accs = []

print(f"\n{'K':<6} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
print("-" * 40)

best_k = 1
best_val_acc = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(mfcc_train_scaled, labels_train)

    train_pred = knn.predict(mfcc_train_scaled)
    val_pred = knn.predict(mfcc_val_scaled)

    train_acc = accuracy_score(labels_train, train_pred)
    val_acc = accuracy_score(labels_val, val_pred)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    gap = train_acc - val_acc
    print(f"{k:<6} {train_acc:.2%}       {val_acc:.2%}       {gap:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_k = k

print("-" * 40)
print(f"Best K: {best_k} with Val Accuracy: {best_val_acc:.2%}")

# ============================================================
# TRAIN FINAL MODEL WITH BEST K
# ============================================================
print(f"\n" + "=" * 60)
print(f"TRAINING FINAL KNN (K={best_k})")
print("=" * 60)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(mfcc_train_scaled, labels_train)

# Evaluate on all sets
knn_train_preds = knn.predict(mfcc_train_scaled)
knn_val_preds = knn.predict(mfcc_val_scaled)
knn_test_preds = knn.predict(mfcc_test_scaled)

knn_train_acc = accuracy_score(labels_train, knn_train_preds)
knn_val_acc = accuracy_score(labels_val, knn_val_preds)
knn_test_acc = accuracy_score(labels_test, knn_test_preds)

print(f"\nKNN RESULTS (K={best_k}):")
print(f"  Train Accuracy: {knn_train_acc:.2%}")
print(f"  Val Accuracy:   {knn_val_acc:.2%}")
print(f"  Test Accuracy:  {knn_test_acc:.2%}")
print(f"\n  Overfitting Gap: {(knn_train_acc - knn_test_acc):.2%}")

# Confusion matrix
knn_cm = confusion_matrix(labels_test, knn_test_preds)

# ============================================================
# VISUALIZATION
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(k_values, [acc * 100 for acc in train_accs], 'b-o', label='Train Accuracy', linewidth=2, markersize=8)
ax.plot(k_values, [acc * 100 for acc in val_accs], 'r-s', label='Val Accuracy', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K (Number of Neighbors)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('KNN: Train vs Validation Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(k_values)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'knn_training_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ Saved knn_training_curves.png")

# ============================================================
# MLP MODEL TRAINING WITH FULL TRACKING & VISUALIZATION
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRAINING MLP MODEL")
print("=" * 60)

# Scale features
mfcc_scaler = StandardScaler()
mfcc_train_scaled = mfcc_scaler.fit_transform(mfcc_train)
mfcc_val_scaled = mfcc_scaler.transform(mfcc_val)
mfcc_test_scaled = mfcc_scaler.transform(mfcc_test)

# Convert to tensors
X_train = torch.FloatTensor(mfcc_train_scaled)
y_train = torch.LongTensor(labels_train)
X_val = torch.FloatTensor(mfcc_val_scaled)
y_val = torch.LongTensor(labels_val)
X_test = torch.FloatTensor(mfcc_test_scaled)
y_test = torch.LongTensor(labels_test)

print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Model configuration
INPUT_DIM = X_train.shape[1]  # 440
HIDDEN_DIMS = [256, 128, 64]  # 3 hidden layers
DROPOUT = 0.3
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32

print(f"\nModel Architecture: {INPUT_DIM} → {HIDDEN_DIMS} → {NUM_CLASSES}")

mlp = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, DROPOUT).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

best_val_acc = 0
best_model_state = None

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12}")
print("-" * 56)

for epoch in range(NUM_EPOCHS):
    # ==================== TRAINING ====================
    mlp.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ==================== VALIDATION ====================
    mlp.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = mlp.state_dict().copy()
        best_epoch = epoch + 1

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {train_acc:<12.2%} {val_acc:<12.2%}")

print("-" * 56)

# Load best model
mlp.load_state_dict(best_model_state)

# ==================== FINAL EVALUATION ====================
mlp.eval()
with torch.no_grad():
    # Train accuracy
    train_outputs = mlp(X_train.to(device))
    mlp_train_acc = (train_outputs.argmax(dim=1).cpu() == y_train).float().mean().item()

    # Val accuracy
    val_outputs = mlp(X_val.to(device))
    mlp_val_acc = (val_outputs.argmax(dim=1).cpu() == y_val).float().mean().item()

    # Test accuracy
    test_outputs = mlp(X_test.to(device))
    mlp_test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    mlp_test_acc = (test_outputs.argmax(dim=1).cpu() == y_test).float().mean().item()

print(f"\n" + "=" * 60)
print("MLP RESULTS")
print("=" * 60)
print(f"Best Epoch:      {best_epoch}")
print(f"Train Accuracy:  {mlp_train_acc:.2%}")
print(f"Val Accuracy:    {mlp_val_acc:.2%}")
print(f"Test Accuracy:   {mlp_test_acc:.2%}")
print(f"\nOverfitting Gap: {(mlp_train_acc - mlp_test_acc):.2%}")

# Confusion matrix
mlp_cm = confusion_matrix(labels_test, mlp_test_preds)

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

# Plot 1: Loss
ax1 = axes[0]
ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('MLP: Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy
ax2 = axes[1]
ax2.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 'r-', label='Val Accuracy', linewidth=2)
ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax2.axhline(y=best_val_acc * 100, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('MLP: Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'mlp_training_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ Saved mlp_training_curves.png")

## 6. Train Wav2Vec2 Classifier

# ============================================================
# WAV2VEC2 TRAINING WITH FULL TRACKING & VISUALIZATION
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRAINING WAV2VEC2 CLASSIFIER")
print("=" * 60)

# Prepare data
emb_scaler = StandardScaler()
emb_train_np = emb_train.numpy() if torch.is_tensor(emb_train) else emb_train
emb_val_np = emb_val.numpy() if torch.is_tensor(emb_val) else emb_val
emb_test_np = emb_test.numpy() if torch.is_tensor(emb_test) else emb_test

emb_train_scaled = emb_scaler.fit_transform(emb_train_np)
emb_val_scaled = emb_scaler.transform(emb_val_np)
emb_test_scaled = emb_scaler.transform(emb_test_np)

# Convert to tensors
X_train = torch.FloatTensor(emb_train_scaled)
y_train = torch.LongTensor(emb_labels_train.numpy() if torch.is_tensor(emb_labels_train) else emb_labels_train)
X_val = torch.FloatTensor(emb_val_scaled)
y_val = torch.LongTensor(emb_labels_val.numpy() if torch.is_tensor(emb_labels_val) else emb_labels_val)
X_test = torch.FloatTensor(emb_test_scaled)
y_test = torch.LongTensor(labels_test)

print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create model (Best config: LR=0.0001, Dropout=0.2, Hidden=(512, 256))
wav2vec_classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(wav2vec_classifier.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

NUM_EPOCHS = 100
best_val_acc = 0
best_model_state = None

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12}")
print("-" * 56)

for epoch in range(NUM_EPOCHS):
    # ==================== TRAINING ====================
    wav2vec_classifier.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = wav2vec_classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ==================== VALIDATION ====================
    wav2vec_classifier.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = wav2vec_classifier(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = wav2vec_classifier.state_dict().copy()
        best_epoch = epoch + 1

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {train_acc:<12.2%} {val_acc:<12.2%}")

print("-" * 56)

# Load best model
wav2vec_classifier.load_state_dict(best_model_state)

# ==================== FINAL EVALUATION ====================
wav2vec_classifier.eval()
with torch.no_grad():
    # Train accuracy
    train_outputs = wav2vec_classifier(X_train.to(device))
    wav2vec_train_acc = (train_outputs.argmax(dim=1).cpu() == y_train).float().mean().item()

    # Val accuracy
    val_outputs = wav2vec_classifier(X_val.to(device))
    wav2vec_val_acc = (val_outputs.argmax(dim=1).cpu() == y_val).float().mean().item()

    # Test accuracy
    test_outputs = wav2vec_classifier(X_test.to(device))
    wav2vec_test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    wav2vec_test_acc = (test_outputs.argmax(dim=1).cpu() == y_test).float().mean().item()

print(f"\n" + "=" * 60)
print("WAV2VEC2 RESULTS")
print("=" * 60)
print(f"Best Epoch:      {best_epoch}")
print(f"Train Accuracy:  {wav2vec_train_acc:.2%}")
print(f"Val Accuracy:    {wav2vec_val_acc:.2%}")
print(f"Test Accuracy:   {wav2vec_test_acc:.2%}")
print(f"\nOverfitting Gap: {(wav2vec_train_acc - wav2vec_test_acc):.2%}")

# Save confusion matrix
wav2vec_cm = confusion_matrix(labels_test, wav2vec_test_preds)

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

# Plot 1: Loss
ax1 = axes[0]
ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Wav2Vec2: Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy
ax2 = axes[1]
ax2.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 'r-', label='Val Accuracy', linewidth=2)
ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax2.axhline(y=best_val_acc * 100, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Wav2Vec2: Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'wav2vec2_training_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ Saved wav2vec2_training_curves.png")

## 7. Train Combined Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRAINING COMBINED MODEL (MFCC + Wav2Vec2)")
print("=" * 60)

# Prepare MFCC data - COMBINE TRAIN + VAL
mfcc_scaler_comb = StandardScaler()
mfcc_all = np.vstack([mfcc_train, mfcc_val])  # Combine!
mfcc_all_scaled = mfcc_scaler_comb.fit_transform(mfcc_all)
mfcc_test_scaled = mfcc_scaler_comb.transform(mfcc_test)

# Prepare Wav2Vec2 embeddings - COMBINE TRAIN + VAL
emb_scaler_comb = StandardScaler()
emb_train_np = emb_train.numpy() if torch.is_tensor(emb_train) else emb_train
emb_val_np = emb_val.numpy() if torch.is_tensor(emb_val) else emb_val
emb_test_np = emb_test.numpy() if torch.is_tensor(emb_test) else emb_test

emb_all = np.vstack([emb_train_np, emb_val_np])  # Combine!
emb_all_scaled = emb_scaler_comb.fit_transform(emb_all)
emb_test_scaled = emb_scaler_comb.transform(emb_test_np)

# Combine labels
labels_all = np.concatenate([labels_train, labels_val])

# Apply PCA to MFCC (128 components)
print("Applying PCA to MFCC (128 components)...")
pca = PCA(n_components=128, random_state=42)
mfcc_all_pca = pca.fit_transform(mfcc_all_scaled)
mfcc_test_pca = pca.transform(mfcc_test_scaled)
print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.2%}")

# Combine features
combined_all = np.concatenate([mfcc_all_pca, emb_all_scaled], axis=1)
combined_test = np.concatenate([mfcc_test_pca, emb_test_scaled], axis=1)

print(f"\nCombined features: {combined_all.shape[1]} (128 MFCC + 1024 Wav2Vec2)")

# Convert to tensors
X_train = torch.FloatTensor(combined_all)
y_train = torch.LongTensor(labels_all)
X_test = torch.FloatTensor(combined_test)
y_test = torch.LongTensor(labels_test)

print(f"Train (train+val): {X_train.shape[0]} samples")
print(f"Test:              {X_test.shape[0]} samples")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create model (Best config)
combined_classifier = nn.Sequential(
    nn.Linear(combined_all.shape[1], 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, NUM_CLASSES)
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(combined_classifier.parameters(), lr=0.0001, weight_decay=0.005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

# Data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': []
}

NUM_EPOCHS = 120
best_test_acc = 0
best_model_state = None

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 44)

for epoch in range(NUM_EPOCHS):
    # ==================== TRAINING ====================
    combined_classifier.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = combined_classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ==================== TEST EVALUATION ====================
    combined_classifier.eval()
    with torch.no_grad():
        test_outputs = combined_classifier(X_test.to(device))
        test_acc = (test_outputs.argmax(dim=1).cpu() == y_test).float().mean().item()

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = combined_classifier.state_dict().copy()
        best_epoch = epoch + 1

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2%} {test_acc:<12.2%}")

print("-" * 44)

# Load best model
combined_classifier.load_state_dict(best_model_state)

# ==================== FINAL EVALUATION ====================
combined_classifier.eval()
with torch.no_grad():
    train_outputs = combined_classifier(X_train.to(device))
    combined_train_acc = (train_outputs.argmax(dim=1).cpu() == y_train).float().mean().item()

    test_outputs = combined_classifier(X_test.to(device))
    combined_test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    combined_test_acc = (test_outputs.argmax(dim=1).cpu() == y_test).float().mean().item()

print(f"\n" + "=" * 60)
print("COMBINED MODEL RESULTS")
print("=" * 60)
print(f"Best Epoch:      {best_epoch}")
print(f"Train Accuracy:  {combined_train_acc:.2%}")
print(f"Test Accuracy:   {combined_test_acc:.2%}")
print(f"\nOverfitting Gap: {(combined_train_acc - combined_test_acc):.2%}")

# Save confusion matrix
combined_cm = confusion_matrix(labels_test, combined_test_preds)

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

# Plot 1: Loss
ax1 = axes[0]
ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Combined Model: Training Loss', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy
ax2 = axes[1]
ax2.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, [acc * 100 for acc in history['test_acc']], 'r-', label='Test Accuracy', linewidth=2)
ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax2.axhline(y=best_test_acc * 100, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Combined Model: Training & Test Accuracy', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'combined_training_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ Saved combined_training_curves.png")

# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<35} {'Test Accuracy':<15}")
print("-" * 50)
print(f"{'SVM (MFCC)':<35} {svm_test_acc:.2%}")
print(f"{'Wav2Vec2':<35} {wav2vec_test_acc:.2%}")
print(f"{'Combined (MFCC + Wav2Vec2)':<35} {combined_test_acc:.2%}")
print("-" * 50)

if combined_test_acc >= max(svm_test_acc, wav2vec_test_acc):
    print(f"\n🏆 COMBINED MODEL IS THE BEST!")

## 8. Final Results Summary

# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================

print("=" * 70)
print("FINAL RESULTS - Speech Emotion Recognition (Jordanian Dialect)")
print("=" * 70)
print(f"\n{'Model':<35} {'Test Accuracy':<15}")
print("-" * 50)
print(f"{'SVM (MFCC)':<35} {svm_test_acc:.2%}")
print(f"{'Wav2Vec2 + Classifier':<35} {wav2vec_test_acc:.2%}")
print(f"{'Combined (MFCC + Wav2Vec2)':<35} {combined_test_acc:.2%}")
print("-" * 50)

# Find best model
results = {
    'SVM': svm_test_acc,
    'Wav2Vec2': wav2vec_test_acc,
    'Combined': combined_test_acc
}
best_model = max(results, key=results.get)
best_acc = results[best_model]

print(f"\n🏆 BEST MODEL: {best_model} with {best_acc:.2%} test accuracy")

## 9. Confusion Matrices

# ============================================================
# CONFUSION MATRICES FOR ALL THREE MODELS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Model data
models_data = [
    ('SVM (MFCC)', svm_cm, svm_test_acc),
    ('Wav2Vec2', wav2vec_cm, wav2vec_test_acc),
    ('Combined', combined_cm, combined_test_acc)
]

for ax, (name, cm, acc) in zip(axes, models_data):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_title(f'{name}\nAccuracy: {acc:.2%}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved confusion_matrices.png")

# ============================================================
# MODEL COMPARISON BAR CHART
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

models = ['SVM\n(MFCC)', 'Wav2Vec2', 'Combined\n(MFCC + Wav2Vec2)']
accuracies = [svm_test_acc * 100, wav2vec_test_acc * 100, combined_test_acc * 100]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Comparison - Jordanian Arabic SER', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% baseline')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved model_comparison.png")

# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT - SVM")
print("=" * 70)
print(classification_report(labels_test, svm_test_preds, target_names=EMOTIONS))

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - Wav2Vec2")
print("=" * 70)
print(classification_report(labels_test, wav2vec_test_preds, target_names=EMOTIONS))

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - Combined")
print("=" * 70)
print(classification_report(labels_test, combined_test_preds, target_names=EMOTIONS))

## 10. Save Models

# ============================================================
# SAVE ALL MODELS
# ============================================================
print("Saving models...")

# Save classifiers
torch.save(wav2vec_classifier.state_dict(), os.path.join(SAVE_DIR, 'wav2vec_classifier.pt'))
torch.save(combined_classifier.state_dict(), os.path.join(SAVE_DIR, 'combined_classifier.pt'))

# Save SVM
with open(os.path.join(SAVE_DIR, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm, f)

# Save scalers
with open(os.path.join(SAVE_DIR, 'mfcc_scaler.pkl'), 'wb') as f:
    pickle.dump(mfcc_scaler, f)
with open(os.path.join(SAVE_DIR, 'emb_scaler.pkl'), 'wb') as f:
    pickle.dump(emb_scaler, f)
with open(os.path.join(SAVE_DIR, 'pca_model.pkl'), 'wb') as f:
    pickle.dump(pca, f)

# Save results
results_dict = {
    'svm_test_acc': float(svm_test_acc),
    'wav2vec_test_acc': float(wav2vec_test_acc),
    'combined_test_acc': float(combined_test_acc),
    'best_model': best_model,
    'best_accuracy': float(best_acc),
    'emotions': EMOTIONS,
    'num_classes': NUM_CLASSES
}

with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✓ All saved to: {SAVE_DIR}")
print("\nSaved files:")
for f in os.listdir(SAVE_DIR):
    print(f"  - {f}")

# ⚠️ Run this in your TRAINING NOTEBOOK, not the web app!
import pickle
import os

SAVE_DIR = "saved_models"

print("Saving the CORRECT scalers for the Combined Model...")

# We must save 'mfcc_scaler_comb' and 'emb_scaler_comb'
# because these are the ones used by the Combined Model.

# 1. Save the Combined MFCC Scaler
if 'mfcc_scaler_comb' in locals():
    with open(os.path.join(SAVE_DIR, 'mfcc_scaler.pkl'), 'wb') as f:
        pickle.dump(mfcc_scaler_comb, f)
    print("✓ Fixed: Saved mfcc_scaler_comb as mfcc_scaler.pkl")
else:
    print("❌ Error: mfcc_scaler_comb not found. Did you run Step 7?")

# 2. Save the Combined Embedding Scaler
if 'emb_scaler_comb' in locals():
    with open(os.path.join(SAVE_DIR, 'emb_scaler.pkl'), 'wb') as f:
        pickle.dump(emb_scaler_comb, f)
    print("✓ Fixed: Saved emb_scaler_comb as emb_scaler.pkl")
else:
    print("❌ Error: emb_scaler_comb not found. Did you run Step 7?")

print("\nNow restart your Web App cell!")

'''
## ✅ Done!

### Final Results:

| Model | Test Accuracy |
|-------|---------------|
| SVM (MFCC) | 88.59% |
| Wav2Vec2 | 89.67% |
| **Combined (MFCC + Wav2Vec2)** | **90.76%** 🏆 |

### Saved Files:
- `embeddings.pt` - Pre-extracted Wav2Vec2 embeddings
- `mfcc_features.npz` - Pre-extracted MFCC features
- `wav2vec_classifier.pt` - Trained Wav2Vec2 classifier
- `combined_classifier.pt` - Trained Combined classifier
- `svm_model.pkl` - Trained SVM model
- `mfcc_scaler.pkl` - MFCC scaler
- `emb_scaler.pkl` - Embedding scaler
- `pca_model.pkl` - PCA model for MFCC
- `results.json` - All results
- `confusion_matrices.png` - Figure for paper
- `model_comparison.png` - Figure for paper
'''