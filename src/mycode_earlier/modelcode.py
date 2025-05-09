import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
import random
import math
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 16000
WINDOW_SIZE = 25  # in ms
HOP_LENGTH = 10   # in ms
N_MFCC = 40
EMB_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANGUAGES = ['hindi', 'english', 'marathi']

# ----------------------------------------- #
# Dataset and Data Augmentation             #
# ----------------------------------------- #

def load_audio(audio_path, target_sr=SAMPLE_RATE):
    """Load audio file and resample if necessary."""
    try:
        if audio_path.endswith('.mp3'):
            y, sr = librosa.load(audio_path, sr=target_sr)
        else:
            y, sr = sf.read(audio_path)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        return y, target_sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None

def normalize_audio(audio):
    """Normalize audio to the range [-1, 1]."""
    return audio / (np.max(np.abs(audio)) + 1e-8)

def augment_audio(audio, sr=SAMPLE_RATE, augment_type=None, noise_var=0.005, pitch_shift=2, time_stretch=None):
    """Apply various augmentations to audio signal."""
    if augment_type is None:
        augment_type = random.choice(['noise', 'pitch', 'stretch', 'speed', None])
    
    if augment_type == 'noise':
        # Add Gaussian noise with variance 0.005
        noise = np.random.normal(0, np.sqrt(noise_var), len(audio))
        return audio + noise
    
    elif augment_type == 'pitch':
        # Shift pitch by ±2 semitones
        shift = random.uniform(-pitch_shift, pitch_shift)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
    
    elif augment_type == 'stretch':
        # Time stretching by factor in [0.9, 1.1]
        if time_stretch is None:
            stretch_factor = random.uniform(0.9, 1.1)
        else:
            stretch_factor = time_stretch
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    elif augment_type == 'speed':
        # Speed perturbation by ±10%
        speed_factor = random.uniform(0.9, 1.1)
        indices = np.round(np.arange(0, len(audio), speed_factor)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]
    
    # No augmentation
    return audio

def extract_features(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features and their deltas from audio."""
    # Convert from ms to samples for window and hop
    win_length = int(WINDOW_SIZE / 1000 * sr)
    hop_length = int(HOP_LENGTH / 1000 * sr)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=win_length,
        hop_length=hop_length
    )
    
    # Get delta and delta-delta (1st and 2nd derivatives)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack all features (n_mfcc * 3, time)
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    # Transpose to (time, n_mfcc * 3)
    features = features.T
    
    return features

def segment_audio(audio, sr=SAMPLE_RATE, window_size=4, hop_size=1):
    """Segment audio into windows with overlap."""
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    # Pad audio if it's shorter than the window size
    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)))
    
    segments = []
    for start in range(0, max(1, len(audio) - window_samples + 1), hop_samples):
        end = start + window_samples
        if end <= len(audio):
            segment = audio[start:end]
            segments.append(segment)
    
    return segments

class KeywordSpottingDataset(Dataset):
    """Dataset for multilingual keyword spotting."""
    
    def __init__(self, audio_files, labels=None, timestamps=None, transform=True, keyword_embeddings=None):
        """
        Initialize KeywordSpottingDataset.
        
        Args:
            audio_files: List of audio file paths
            labels: List of labels (1 for keyword, 0 for no keyword)
            timestamps: List of (start, end) timestamps for keywords
            transform: Whether to apply augmentation
            keyword_embeddings: Dictionary of pre-computed keyword embeddings
        """
        self.audio_files = audio_files
        self.labels = labels if labels is not None else [0] * len(audio_files)
        self.timestamps = timestamps if timestamps is not None else [(0, 0)] * len(audio_files)
        self.transform = transform
        self.keyword_embeddings = keyword_embeddings or {}
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        timestamp = self.timestamps[idx]
        
        # Load and preprocess audio
        audio, sr = load_audio(audio_path)
        
        if audio is None:
            # Provide a fallback if audio loading fails
            audio = np.zeros(int(SAMPLE_RATE * 4))  # 4 seconds of silence
            sr = SAMPLE_RATE
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Apply augmentation if required
        if self.transform:
            audio = augment_audio(audio, sr)
        
        # Extract features
        features = extract_features(audio, sr)
        
        # Create target sequence with 1s at keyword positions
        if label == 1 and timestamp != (0, 0):
            # Convert timestamp to frame indices
            start_frame = int(timestamp[0] * sr / (HOP_LENGTH / 1000 * sr))
            end_frame = int(timestamp[1] * sr / (HOP_LENGTH / 1000 * sr))
            
            # Ensure frames are within the feature sequence
            start_frame = max(0, min(start_frame, features.shape[0] - 1))
            end_frame = max(0, min(end_frame, features.shape[0] - 1))
            
            # Create target sequence
            target = np.zeros(features.shape[0])
            target[start_frame:end_frame+1] = 1
        else:
            target = np.zeros(features.shape[0])
        
        return {
            'features': torch.FloatTensor(features),
            'target': torch.FloatTensor(target),
            'label': label,
            'path': audio_path,
            'timestamp': timestamp
        }

# ----------------------------------------- #
# Model Architecture                        #
# ----------------------------------------- #

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class CNNKeywordEmbedding(nn.Module):
    """CNN model for generating keyword embeddings."""
    
    def __init__(self, input_dim=N_MFCC*3, emb_size=EMB_SIZE):
        super(CNNKeywordEmbedding, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, emb_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        Returns:
            Embedding vector [batch_size, emb_size]
        """
        # Transpose for 1D convolution [batch, channels, length]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Final embedding
        x = self.fc(x)
        
        # Normalize embedding to unit length
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        
        return x

class FSLAKWS(nn.Module):
    """Few-Shot Language Agnostic Keyword Spotting System."""
    
    def __init__(self, input_dim=N_MFCC*3, hidden_dim=256, num_layers=4, num_heads=4, ff_dim=512, dropout=0.1):
        super(FSLAKWS, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, EMB_SIZE)
    
    def forward(self, x, src_mask=None):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            src_mask: Mask for transformer (for padding)
        Returns:
            Sequence of embeddings [batch_size, seq_len, emb_size]
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # Project to embedding size
        x = self.output_projection(x)
        
        # Normalize embeddings
        x = x / (torch.norm(x, dim=2, keepdim=True) + 1e-8)
        
        return x

# ----------------------------------------- #
# Training and Evaluation Functions         #
# ----------------------------------------- #

def generate_keyword_embeddings(keyword_files, cnn_model, device=DEVICE):
    """
    Generate embeddings for keywords using the CNN model.
    
    Args:
        keyword_files: Dictionary mapping keywords to list of audio files
        cnn_model: Trained CNN model for embedding extraction
        device: Device to run the model on
    
    Returns:
        Dictionary mapping keywords to their embeddings
    """
    embeddings = {}
    cnn_model.eval()
    
    with torch.no_grad():
        for keyword, files in keyword_files.items():
            keyword_embeddings = []
            
            for file_path in files:
                # Load and preprocess audio
                audio, sr = load_audio(file_path)
                if audio is None:
                    continue
                
                audio = normalize_audio(audio)
                features = extract_features(audio, sr)
                
                # Convert to tensor and add batch dimension
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                
                # Get embedding
                embedding = cnn_model(features_tensor)
                keyword_embeddings.append(embedding.cpu().numpy())
            
            if keyword_embeddings:
                # Average embeddings for this keyword
                embeddings[keyword] = np.mean(np.vstack(keyword_embeddings), axis=0)
    
    return embeddings

def train_embedding_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device=DEVICE):
    """
    Train the CNN embedding model using cosine similarity loss.
    
    Args:
        model: CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained model and training history
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Cosine similarity loss with margin
    def cosine_similarity_loss(embeddings1, embeddings2, targets, margin=0.2):
        # Compute cosine similarity
        cos_sim = torch.sum(embeddings1 * embeddings2, dim=1)
        
        # Loss: pull positives together, push negatives apart
        pos_loss = (1 - cos_sim) * targets
        neg_loss = torch.clamp(cos_sim - margin, min=0.0) * (1 - targets)
        
        return torch.mean(pos_loss + neg_loss)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Create positive and negative pairs
            # For simplicity, we use the same batch as positives and negatives
            embeddings = model(features)
            
            # Roll the embeddings to create pairs
            embeddings_shifted = torch.roll(embeddings, 1, dims=0)
            
            # Create targets (1 for same keyword, 0 for different)
            labels = (batch['label'] == torch.roll(batch['label'], 1, dims=0)).float().to(device)
            
            # Compute loss
            loss = cosine_similarity_loss(embeddings, embeddings_shifted, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                features = batch['features'].to(device)
                
                # Create positive and negative pairs
                embeddings = model(features)
                embeddings_shifted = torch.roll(embeddings, 1, dims=0)
                
                # Create targets
                labels = (batch['label'] == torch.roll(batch['label'], 1, dims=0)).float().to(device)
                
                # Compute loss
                loss = cosine_similarity_loss(embeddings, embeddings_shifted, labels)
                val_loss += loss.item()
        
        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, history

def train_transformer_model(model, train_loader, val_loader, keyword_embeddings, 
                           num_epochs=10, lr=0.001, device=DEVICE):
    """
    Train the transformer model for keyword spotting.
    
    Args:
        model: Transformer model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        keyword_embeddings: Dictionary of pre-computed keyword embeddings
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained model and training history
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Binary cross-entropy loss for sequence detection
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Convert keyword embeddings to tensors and send to device
    keyword_embs = {k: torch.FloatTensor(v).to(device) for k, v in keyword_embeddings.items()}
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass through transformer
            speech_embs = model(features)
            
            # For each batch item, find the closest keyword
            batch_size, seq_len, emb_dim = speech_embs.shape
            
            # Initialize predictions
            preds = torch.zeros(batch_size, seq_len).to(device)
            
            # Compute cosine similarity with each keyword
            for i in range(batch_size):
                max_sim = torch.zeros(seq_len).to(device)
                
                # For each frame, compute similarity with all keywords
                for keyword, keyword_emb in keyword_embs.items():
                    # Reshape keyword embedding for broadcasting
                    kw_emb = keyword_emb.view(1, -1).expand(seq_len, -1)
                    
                    # Compute similarity
                    sim = torch.sum(speech_embs[i] * kw_emb, dim=1)
                    
                    # Update max similarity
                    max_sim = torch.maximum(max_sim, sim)
                
                preds[i] = max_sim
            
            # Compute BCE loss
            loss = bce_loss(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_preds.extend((preds > 0.5).float().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_precision, train_recall, train_f1 = calculate_metrics(all_preds, all_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                
                # Forward pass
                speech_embs = model(features)
                
                # Initialize predictions
                batch_size, seq_len, emb_dim = speech_embs.shape
                preds = torch.zeros(batch_size, seq_len).to(device)
                
                # Compute cosine similarity with each keyword
                for i in range(batch_size):
                    max_sim = torch.zeros(seq_len).to(device)
                    
                    for keyword, keyword_emb in keyword_embs.items():
                        kw_emb = keyword_emb.view(1, -1).expand(seq_len, -1)
                        sim = torch.sum(speech_embs[i] * kw_emb, dim=1)
                        max_sim = torch.maximum(max_sim, sim)
                    
                    preds[i] = max_sim
                
                # Compute loss
                loss = bce_loss(preds, targets)
                val_loss += loss.item()
                
                # Store predictions and targets
                all_preds.extend((preds > 0.5).float().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        val_precision, val_recall, val_f1 = calculate_metrics(all_preds, all_targets)
        
        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, "
              f"Val F1: {val_f1:.4f}")
    
    return model, history

def calculate_metrics(predictions, targets):
    """Calculate precision, recall, and F1 score."""
    predictions = np.array(predictions) > 0.5
    targets = np.array(targets) > 0.5
    
    true_positives = np.sum(predictions & targets)
    false_positives = np.sum(predictions & ~targets)
    false_negatives = np.sum(~predictions & targets)
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

def evaluate_model(model, test_loader, keyword_embeddings, threshold=0.8, device=DEVICE):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained transformer model
        test_loader: DataLoader for test data
        keyword_embeddings: Dictionary of pre-computed keyword embeddings
        threshold: Detection threshold for similarity
        device: Device to run evaluation on
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Convert keyword embeddings to tensors
    keyword_embs = {k: torch.FloatTensor(v).to(device) for k, v in keyword_embeddings.items()}
    
    all_preds = []
    all_targets = []
    all_timestamps = []
    all_pred_timestamps = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            true_timestamps = batch['timestamp']
            
            # Forward pass
            speech_embs = model(features)
            
            # Initialize predictions
            batch_size, seq_len, emb_dim = speech_embs.shape
            batch_preds = []
            pred_timestamps = []
            
            # For each item in batch
            for i in range(batch_size):
                # Get maximum similarity across all keywords
                max_sim = torch.zeros(seq_len).to(device)
                best_keyword = None
                
                for keyword, keyword_emb in keyword_embs.items():
                    kw_emb = keyword_emb.view(1, -1).expand(seq_len, -1)
                    sim = torch.sum(speech_embs[i] * kw_emb, dim=1)
                    
                    # Update if this keyword has better similarity
                    if torch.max(sim) > torch.max(max_sim):
                        max_sim = sim
                        best_keyword = keyword
                
                # Apply temporal smoothing (5-frame moving average)
                smooth_sim = torch.zeros_like(max_sim)
                for t in range(seq_len):
                    start = max(0, t - 2)
                    end = min(seq_len, t + 3)
                    smooth_sim[t] = torch.mean(max_sim[start:end])
                
                # Apply threshold
                pred = (smooth_sim > threshold).float()
                batch_preds.append(pred.cpu().numpy())
                
                # Convert frame indices to timestamps
                if torch.any(pred > 0.5):
                    frames = torch.where(pred > 0.5)[0].cpu().numpy()
                    
                    # Group consecutive frames
                    groups = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
                    
                    # Convert groups to timestamps
                    for group in groups:
                        if len(group) > 0:
                            start_time = group[0] * (HOP_LENGTH / 1000)
                            end_time = group[-1] * (HOP_LENGTH / 1000)
                            pred_timestamps.append((start_time, end_time, best_keyword))
                else:
                    pred_timestamps.append((0, 0, None))
            
            all_preds.extend(batch_preds)
            all_targets.extend(targets.cpu().numpy())
            all_timestamps.extend(true_timestamps)
            all_pred_timestamps.extend(pred_timestamps)
    
    # Calculate metrics
    precision, recall, f1 = calculate_metrics(all_preds, all_targets)
    
    # Calculate timestamp error
    timestamp_errors = []
    for true_ts, pred_ts in zip(all_timestamps, all_pred_timestamps):
        if true_ts != (0, 0) and pred_ts != (0, 0, None):
            # Calculate start and end errors
            start_error = abs(true_ts[0] - pred_ts[0])
            end_error = abs(true_ts[1] - pred_ts[1])
            timestamp_errors.append((start_error + end_error) / 2)
    
    avg_ts_error = np.mean(timestamp_errors) if timestamp_errors else float('inf')
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'timestamp_error': avg_ts_error,
        'predictions': all_preds,
        'targets': all_targets,
        'true_timestamps': all_timestamps,
        'pred_timestamps': all_pred_timestamps
    }
    
    return results

def detect_keyword_in_audio(audio_path, model, keyword_embeddings, threshold=0.8, device=DEVICE):
    """
    Detect keywords in a single audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained transformer model
        keyword_embeddings: Dictionary of pre-computed keyword embeddings
        threshold: Detection threshold for similarity
        device: Device to run detection on
    
    Returns:
        List of (start_time, end_time, keyword) tuples
    """
    model.eval()
    
    # Convert keyword embeddings to tensors
    keyword_embs = {k: torch.FloatTensor(v).to(device) for k, v in keyword_embeddings.items()}
    
    # Load and preprocess audio
    audio, sr = load_audio(audio_path)
    if audio is None:
        return []
    
    audio = normalize_audio(audio)
    
    # Segment audio into 4-second chunks with 1-second overlap
    segments = segment_audio(audio, sr)
    
    all_detections = []
    
    with torch.no_grad():
        for i, segment in enumerate(segments):
            # Extract features
            features = extract_features(segment, sr)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            # Get speech embeddings
            speech_embs = model(features_tensor)
            
            # Initialize variables for detection
            seq_len = speech_embs.shape[1]
            max_sim = torch.zeros(seq_len).to(device)
            best_keyword = None
            
            # Compare with all keyword embeddings
            for keyword, keyword_emb in keyword_embs.items():
                kw_emb = keyword_emb.view(1, -1).expand(seq_len, -1)
                sim = torch.sum(speech_embs[0] * kw_emb, dim=1)
                
                # Update if this keyword has better similarity
                if torch.max(sim) > torch.max(max_sim):
                    max_sim = sim
                    best_keyword = keyword
            
            # Apply temporal smoothing (5-frame moving average)
            smooth_sim = torch.zeros_like(max_sim)
            for t in range(seq_len):
                start = max(0, t - 2)
                end = min(seq_len, t + 3)
                smooth_sim[t] = torch.mean(max_sim[start:end])
            
            # Apply threshold and find detections
            pred = (smooth_sim > threshold).float()
            
            if torch.any(pred > 0.5):
                frames = torch.where(pred > 0.5)[0].cpu().numpy()
                
                # Group consecutive frames
                groups = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
                
                # Convert groups to timestamps (adjust for segment offset)
                segment_offset = i * 1  # 1 second hop between segments
                for group in groups:
                    if len(group) > 0:
                        start_time = segment_offset + group[0] * (HOP_LENGTH / 1000)
                        end_time = segment_offset + group[-1] * (HOP_LENGTH / 1000)
                        all_detections.append((start_time, end_time, best_keyword))
    
    # Merge overlapping detections from adjacent segments
    merged_detections = []
    if all_detections:
        # Sort by start time
        all_detections.sort(key=lambda x: x[0])
        
        current_start, current_end, current_kw = all_detections[0]
        
        for start, end, kw in all_detections[1:]:
            if start <= current_end and kw == current_kw:
                # Merge with current detection
                current_end = max(current_end, end)
            else:
                # Add current detection to results
                merged_detections.append((current_start, current_end, current_kw))
                current_start, current_end, current_kw = start, end, kw
        
        # Add the last detection
        merged_detections.append((current_start, current_end, current_kw))
    
    return merged_detections

# ----------------------------------------- #
# Main Training and Evaluation Pipeline     #
# ----------------------------------------- #

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load dataset (replace with your actual data loading)
    print("Loading dataset...")
    
    # For demonstration, we'll create mock data
    # In practice, you would load your actual dataset here
    audio_files = []
    labels = []
    timestamps = []
    
    # Create mock data for each language
    for lang in LANGUAGES:
        for i in range(50):  # 50 samples per language
            audio_files.append(f"data/{lang}/sample_{i}.wav")
            # Randomly assign labels (1 for keyword, 0 for non-keyword)
            label = random.choice([0, 1])
            labels.append(label)
            
            # For keyword samples, assign random timestamps
            if label == 1:
                start = random.uniform(1.0, 3.0)
                end = start + random.uniform(0.5, 1.5)
                timestamps.append((start, end))
            else:
                timestamps.append((0, 0))
    
    # Split into train, validation, and test sets
    train_files, test_files, train_labels, test_labels, train_ts, test_ts = train_test_split(
        audio_files, labels, timestamps, test_size=0.3, random_state=42
    )
    
    val_files, test_files, val_labels, test_labels, val_ts, test_ts = train_test_split(
        test_files, test_labels, test_ts, test_size=0.5, random_state=42
    )
    
    # Create datasets
    train_dataset = KeywordSpottingDataset(train_files, train_labels, train_ts, transform=True)
    val_dataset = KeywordSpottingDataset(val_files, val_labels, val_ts, transform=False)
    test_dataset = KeywordSpottingDataset(test_files, test_labels, test_ts, transform=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Step 1: Train the CNN embedding model
    print("\nTraining CNN embedding model...")
    cnn_model = CNNKeywordEmbedding()
    cnn_model, cnn_history = train_embedding_model(
        cnn_model, train_loader, val_loader, num_epochs=10, lr=0.001
    )
    
    # Step 2: Generate keyword embeddings
    print("\nGenerating keyword embeddings...")
    
    # In practice, you would load your actual keyword samples
    # Here we'll use the training data that has label=1 as keyword samples
    keyword_files = {}
    for file_path, label in zip(train_files, train_labels):
        if label == 1:
            # Extract keyword name from path (mock)
            keyword = os.path.basename(file_path).split('_')[0]
            if keyword not in keyword_files:
                keyword_files[keyword] = []
            keyword_files[keyword].append(file_path)
    
    keyword_embeddings = generate_keyword_embeddings(keyword_files, cnn_model)
    
    # Step 3: Train the transformer model
    print("\nTraining transformer model...")
    transformer_model = FSLAKWS()
    transformer_model, transformer_history = train_transformer_model(
            transformer_model, train_loader, val_loader, keyword_embeddings, 
        num_epochs=15, lr=0.0005
    )

    # Step 4: Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(transformer_model, test_loader, keyword_embeddings)
    
    print("\nTest Results:")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"Average Timestamp Error: {test_results['timestamp_error']:.4f} seconds")

    # Step 5: Example of detecting keywords in new audio
    print("\nRunning detection on example audio...")
    example_audio = "data/example.wav"  # Replace with your test file
    
    if os.path.exists(example_audio):
        detections = detect_keyword_in_audio(example_audio, transformer_model, keyword_embeddings)
        
        print("\nDetected Keywords:")
        for start, end, keyword in detections:
            print(f"{keyword}: {start:.2f}s - {end:.2f}s")
    else:
        print(f"Example audio file {example_audio} not found")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # CNN training curves
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history['train_loss'], label='Train Loss')
    plt.plot(cnn_history['val_loss'], label='Val Loss')
    plt.title('CNN Embedding Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Transformer training curves
    plt.subplot(1, 2, 2)
    plt.plot(transformer_history['train_loss'], label='Train Loss')
    plt.plot(transformer_history['val_loss'], label='Val Loss')
    plt.plot(transformer_history['train_f1'], label='Train F1')
    plt.plot(transformer_history['val_f1'], label='Val F1')
    plt.title('Transformer Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Save models
    torch.save(cnn_model.state_dict(), 'cnn_embedding_model.pth')
    torch.save(transformer_model.state_dict(), 'transformer_model.pth')
    print("\nModels saved to disk.")

if __name__ == "__main__":
    main()