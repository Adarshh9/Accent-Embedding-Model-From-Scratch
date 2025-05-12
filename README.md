# Accent Embedding Model From Scratch

## Repository Overview

This repository contains a comprehensive implementation for extracting accent embeddings from speech data. The system processes raw audio files, extracts various acoustic features, and trains a deep learning model to generate compact embeddings that capture accent characteristics.

## Key Features

- **Audio Preprocessing Pipeline**: Advanced audio processing including resampling, silence trimming, normalization, and filtering
- **Multi-Feature Extraction**: Extracts mel spectrograms, prosody features, formants, MFCCs, and voice quality metrics
- **Deep Learning Model**: Custom transformer-based architecture with:
  - Feature-specific encoders with residual connections
  - Multi-head attention for temporal modeling
  - Multi-task learning with reconstruction objectives
  - Contrastive and triplet loss for discriminative embeddings
- **End-to-End Training**: Complete training loop with validation and model checkpointing

## Model Architecture

The accent embedding model uses a hybrid architecture:

1. **Feature-Specific Encoders**:
   - Residual CNNs for mel spectrograms and prosody features
   - Separate encoders for different spectral features (MFCC, chroma, etc.)

2. **Transformer Encoder**:
   - Captures temporal relationships in acoustic features
   - Positional encoding for sequence information

3. **Fusion Network**:
   - Combines features from different modalities
   - Learns weighted feature importance

4. **Multi-Task Heads**:
   - Embedding generation (main task)
   - Feature reconstruction (auxiliary tasks)

## Dataset

The model is trained on the L2-ARCTIC dataset, which contains:
- 8 non-native English speakers (4 Hindi, 4 Spanish accents)
- Over 1,000 utterances per speaker
- Professional studio recordings

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3 (for GPU acceleration)
- Additional audio processing libraries:
  ```
  librosa
  numpy
  soundfile
  torchaudio
  praat-parselmouth
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Accent-Embedding-Model-From-Scratch.git
   cd Accent-Embedding-Model-From-Scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Feature Extraction**:
   ```python
   processor = FeatureProcessor(
       input_root="/path/to/l2_arctic",
       output_root="/path/to/processed_features"
   )
   processor.process_dataset()
   ```

2. **Model Training**:
   ```python
   model = EnhancedAccentEncoder().to(device)
   train_accent_encoder(model, train_loader, val_loader, num_epochs=50, device=device)
   ```

3. **Embedding Extraction**:
   ```python
   embeddings, labels, accents = extract_embeddings(model, dataloader, device)
   ```

## Results

The model achieves:
- Strong separation between different accent groups in embedding space
- Effective reconstruction of input features
- Robust performance across different speakers within the same accent group

## Applications

These accent embeddings can be used for:
- Accent identification and classification
- Accent conversion systems
- Pronunciation scoring
- Speaker verification with accent awareness
- Linguistic research on accent characteristics

## Future Work

- Expand to more accent groups
- Incorporate multilingual capabilities
- Develop real-time inference capabilities
- Explore self-supervised pre-training
