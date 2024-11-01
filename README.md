# Pitch Classification Using Computer Vision and Audio Analysis

This project classifies musical pitches by leveraging computer vision, mel spectrograms, and transfer learning based on a pre-trained VGGish model. It utilizes synthesized data to augment the dataset, enabling accurate pitch classification even with limited initial data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to classify musical pitches accurately from audio recordings. The project uses mel spectrograms of audio samples, which are processed with computer vision techniques and analyzed using transfer learning from the VGGish model (a CNN pre-trained on audio data). Additionally, the project incorporates data synthesis to expand and diversify the dataset, hoping to improving the robustness and generalization of the model.

## Challenges
The very first challenge lies in translating audio data (in this case .wav-files) into something we can use as features. There are numeral features we could extract from audio data, like the following:

### MFCCs
The so called Mel-Frequency Cepstral Coefficients captures audio data in a way that resembles the human way of hearing, which puts an emphasis on lower frequencies.  
  
Let me try to explain:  
The Mel scale is a perceptual scale of pitches, where each "Mel" unit represents an equally spaced step in perceived pitch. This scale is derived from how humans perceive sound frequencies; we are more sensitive to lower frequencies than higher ones.  
In MFCC computation, the frequency of audio is transformed onto this Mel scale to capture the way humans hear sound.  
This is done via computing the fourier transformation from the original signal, giving us a conversion from the time-domain into the frequency domain resulting in a spectrum of frequencies:  

#### From the original signal:

<div align="center">
    <img src="https://github.com/user-attachments/assets/baab915a-25bb-420d-9388-0e2ec4338487" alt="original signal" width="600"/>
</div>

#### To the fourier transformed frequencies (FFT):

<div align="center">
    <img src="https://github.com/user-attachments/assets/f3362e5e-fc95-46c3-b7ee-7c3b82c49fd6" alt="original signal" width="600"/>
</div>

#### We then compute a spectogram using short, overlapping FFTs, using a sliding window
This shows how the frequencies in the audio change over time.

<div align="center">
    <img src="https://github.com/user-attachments/assets/3fbb77ed-3d0f-4b0a-a8f7-a135d4136ba8" alt="original signal" width="600"/>
</div>

#### Scaled MFCC version with more emphasis on the lower frequencies, where we map the mal-scale onto the y-axis:

<div align="center">
    <figure style="display: inline-block; margin-right: 20px;">
        <img src="https://github.com/user-attachments/assets/928fb030-f6d2-4b86-88bc-d1f5ccb0f08f" alt="Original Signal" width="400"/>
        <figcaption>High frequency resolution</figcaption>
    </figure>
    <figure style="display: inline-block;">
        <img src="https://github.com/user-attachments/assets/33380b38-10bc-473f-91f4-dc8216dd65b3" alt="Transformed Signal" width="400"/>
        <figcaption>Lower frequency resolution</figcaption>
    </figure>
</div>




## Features
- **Audio to Image Transformation**: Audio samples are converted into mel spectrogram images, creating a visual representation of pitch and frequency.
- **Transfer Learning with VGGish**: The VGGish model, pre-trained on general audio data, is fine-tuned for pitch classification tasks.
- **Data Synthesis and Augmentation**: Synthetic data generation is used to expand the dataset, introducing varied conditions that improve model performance and robustness.
- **Pitch Classification**: Model outputs the classified pitch from a given audio sample.

## Dataset
The dataset consists of audio samples transformed into mel spectrograms:
1. **Real Audio Samples**: Collected from available pitch-labeled datasets.
2. **Synthetic Audio Samples**: Generated using audio synthesis to create a broader range of pitches and conditions.

## Data Augmentation
To address the limitations of a small initial dataset, synthesized audio is added. This synthesis includes variations in pitch, duration, and noise levels, aiding in training a model that generalizes well to new audio inputs.

## Model Architecture
The classification pipeline uses VGGish as a feature extractor:
1. **Mel Spectrogram Extraction**: Each audio sample is transformed into a mel spectrogram image.
2. **Feature Extraction**: Using the VGGish model, features from the spectrograms are extracted.
3. **Classification Layer**: A custom dense layer is added on top of the VGGish output for final pitch classification.

## Training
1. **Transfer Learning**: The VGGish model is frozen initially, training only the classifier layers.
2. **Fine-Tuning**: In later stages, the VGGish layers are gradually unfrozen to adapt more closely to pitch recognition.
3. **Data Augmentation**: Both real and synthetic data samples are used, applying techniques like pitch shifting and noise addition to improve model resilience.

