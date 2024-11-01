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
  
The Mel scale divides the frequency domain into bins that are narrower at lower frequencies and wider at higher frequencies. This aligns with human auditory perception: we can easily distinguish between frequencies like 2000 Hz and 2500 Hz, but we struggle to detect differences between, for example, 10000 Hz and 10500 Hz, even though the frequency gap is the same.  

In MFCC computation, the frequency of audio is transformed onto this Mel scale to capture the way humans hear sound.  
This is done via computing the fourier transformation (actual multiple) from the original signal, giving us a conversion from the time-domain into the frequency domain resulting in a spectrum of frequencies, then creating a spectogram and scaling it on the Mel scale:  

#### From the original signal, here a A-Major chord:

<div align="center">
    <img src="https://github.com/user-attachments/assets/baab915a-25bb-420d-9388-0e2ec4338487" alt="original signal" width="600"/>
</div>

#### To the fourier transformed frequencies (FFT):

<div align="center">
    <img src="https://github.com/user-attachments/assets/f3362e5e-fc95-46c3-b7ee-7c3b82c49fd6" alt="original signal" width="600"/>
</div>

#### To the spectogram using short, overlapping FFTs, using a sliding window:
This shows how the frequencies in the audio change over time.

<div align="center">
    <img src="https://github.com/user-attachments/assets/3fbb77ed-3d0f-4b0a-a8f7-a135d4136ba8" alt="original signal" width="600"/>
</div>

And Finally we get the scaled MFCC version with more emphasis on the lower frequencies, where we map the mal-scale onto the y-axis.
On the left we see a higher resolution frequency, using 128 mel-scaled bins. Each bin corresponds wo a specific frequency range on the mel-scale.
On the right we have a lower frequency resolution which is computationally less expensive and could also be used for simpler tasks.

<div align="center">
    <img src="https://github.com/user-attachments/assets/928fb030-f6d2-4b86-88bc-d1f5ccb0f08f" alt="original signal" width="400" style="display: inline-block; margin-right: 10px;"/>
    <img src="https://github.com/user-attachments/assets/33380b38-10bc-473f-91f4-dc8216dd65b3" alt="transformed signal" width="400" style="display: inline-block;"/>
</div>

### Chroma features

Another approach I tried in this project is concerned with chroma features. These can easily be recognized as a pitch and are insensitive to the actual octave the chord is played, which make them a suitable alternative to the Mel spectograms for pitch detection.

In this example, we also use an A-Major chord, and its easiy to spot which pitches are the most dominant: A (the root), C# (major third) and E (perfect fifth) which make up the A-Major chord:

<div align="center">
    <img src="https://github.com/user-attachments/assets/100bd829-7fad-4244-bfae-7a9fed9d33db" alt="original signal" width="600"/>
</div>

There could be additional features which can be used in the future, like spectral contrast or RMS (Root Mean Square).  
Spectral contrast (left) measures the difference in amplitude between peaks and valleys in different frequency bands across time. It essentially highlights regions where the sound spectrum varies significantly within certain frequency bands.  
RMS (Root Mean Square, left) measures the average power or loudness of the signal over time. It provides an understanding of the energy level at each frame, showing variations in loudness across the audio.  


<div align="center">
    <img src="https://github.com/user-attachments/assets/2d9f936f-ab8e-47d4-b19f-cacdfde4e10b" alt="original signal" width="400" style="display: inline-block; margin-right: 10px;"/>
    <img src="https://github.com/user-attachments/assets/26fb08b1-3fd3-4b67-a750-71b2f2a60a89" alt="transformed signal" width="400" style="display: inline-block;"/>
</div>

For this project, I will focus on the first two features: Mel spectograms and chroma features.  

# First iteration
- **Audio to Image Transformation**: Audio samples are converted into mel spectrogram images, creating a visual representation of pitch and frequency.
- **Transfer Learning with VGGish**: The VGGish model, pre-trained on general audio data, is fine-tuned for pitch classification tasks.
- **Pitch Classification**: Model outputs the classified pitch from a given audio sample.

## Dataset
The dataset consists of audio samples, downloaded from kaggle at https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3  
This dataset consists out of seven different chords, namely A-Minor, Bb-Major, C-Major, D-Minor, E-Minor, F-Major and G-Major, and consists of recordings of accoustic as well as electric guitar strums of those chords.  
  
The dataset is considerably small, only having 140 samples per chord in the training set, and 40 samples in the test set.

### Examples:

F-Major:

[▶️ Play F-Major](https://drive.google.com/file/d/1nlIc4Q0YVQoFdVYccCpqwh9jKLc-NPwL/view?usp=sharing)

A-Minor:

[▶️ Play A-Minor](https://drive.google.com/file/d/151oZcGclXDwP9HWnFKk3NelCmvnMDQlH/view?usp=sharing)


## Model Architecture of VGGish
The classification pipeline uses VGGish as a feature extractor.

VGGish is a CNN architecture, which was developed by Google in 2017 and adapted to audio signal classification. It is specifically designed to handle Mel spectograms with a length of 0.96 seconds.  
It consists of the following main building blocks:  

Convolutional Blocks: VGGish consists of multiple convolutional blocks, each comprising:  
- Convolutional layers with 96 filters of size 3x3  
- Batch normalization  
- ReLU activation  
Max Pooling: Max pooling layers with a stride of 2 are used to downsample the feature maps.  
Fully Connected Layers: The output of the convolutional blocks is flattened and fed into fully connected layers with ReLU activation.  
Output Layer: The final output layer has a single neuron with a sigmoid activation function, producing a probability distribution over the 527 AudioSet classes.

This model was trained on: 

<div align="center">
    <img src="https://github.com/user-attachments/assets/77606bc3-f451-42a4-a6d1-ce9ac680a5f7" alt="original signal" width="600"/>
</div>

  
The idea is using it as a convolutional base (a feature extractor) and retraining a custom head:  
  
<div align="center">
    <img src="https://github.com/user-attachments/assets/9121f2f2-ac65-4be2-b7f4-8d73372cd8f8" alt="original signal" width="600"/>
</div>




## Training
1. **Transfer Learning**: The VGGish model is frozen initially, training only the classifier layers.
2. **Fine-Tuning**: In later stages, the VGGish layers are gradually unfrozen to adapt more closely to pitch recognition.
3. **Data Augmentation**: Both real and synthetic data samples are used, applying techniques like pitch shifting and noise addition to improve model resilience.



## Data Augmentation
To address the limitations of a small initial dataset, synthesized audio is added. This synthesis includes variations in pitch, duration, and noise levels, aiding in training a model that generalizes well to new audio inputs.

