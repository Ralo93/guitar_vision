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

##### For the first iteration of this project, I decided to simply train a major/minor distinguisher, as I thought this might be easier for the start and get used to the transfer learning approach. So for the start, we have a binary classification problem here.

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
Output Layer: The final layer should have 527 neurons with a softmax activation function to produce a probability distribution over the 527 classes in AudioSet.

Here we can see what VGGish was trained on: 

<div align="center">
    <img src="https://github.com/user-attachments/assets/77606bc3-f451-42a4-a6d1-ce9ac680a5f7" alt="original signal" width="600"/>
</div>


## Training
0. **Preprocessing the images**: After transforming the audio files into Mel spectograms, I use normalization for the input images, because it smoothens the training curve.
1. **Transfer Learning**: The VGGish model is frozen initially, training only the classifier layers.
2. **Fine-Tuning**: In later stages, the VGGish layers are gradually unfrozen to adapt more closely to pitch recognition.

  
#### Not normalized input vs. normalized input:  
  
<div align="center">
    <img src="https://github.com/user-attachments/assets/1afb41ca-b614-4675-9951-e055aedd14d9" alt="original signal" width="400" style="display: inline-block; margin-right: 10px;"/>
    <img src="https://github.com/user-attachments/assets/9faac160-210e-4511-ab7f-a2e9301cdd7c" alt="transformed signal" width="400" style="display: inline-block;"/>
</div> 

We can see here, that normalizing the inputs to the model "smoothens" the loss curve.

#### Transfer learning  
The custom classifier on top of the VGGish model had 8386 trainable parameters:
  
![image](https://github.com/user-attachments/assets/c7cd09cb-de4c-4858-b03d-95c4fc387964)  

This did only perform around 55% overall accuracy. Slightly better than throwing a coin:

<div align="center">
    <img src="https://github.com/user-attachments/assets/f0c5a40e-c71c-4a53-9f43-10c1106e30a5" alt="original signal" width="400"/>
</div>
  
  
The next step is to now fine-tune the last layers of the VGGish model together with the trained classifier on top:

<div align="center">
🟧 The unfrozen last layer
🟥 The classifier on top  
</div>  
<div align="center">
    <img src="https://github.com/user-attachments/assets/eb5d6b3d-9fee-4c1f-9d31-6846fa959deb" alt="original signal" width="400"/>
</div>

  
### The model was now successively trained on unfrozen last layers:

Unfrozen classifier and unfrozen last layer of the base model:  

<div align="center">
    <img src="https://github.com/user-attachments/assets/b32d9d1c-f69d-4bdf-932d-cc19b8b060e6" alt="original signal" width="400"/>
</div>

Unfrozen classifier and unfrozen two last layers of the base model:  

<div align="center">
    <img src="https://github.com/user-attachments/assets/ffd7fbf6-4c87-43e3-bf4c-504d0605403a" alt="original signal" width="400"/>
</div>

  
We can clearly see that the model starts improving on the validation set, which is great, even though the performance is not that strong. But keep in mind the dataset we work with is extremely small. 

  
## Second Iteration
Now it is time to try out the actual pitch detection. For this I chose to try out the chroma features of the different pitches, and for the beginning, a simple CNN with this architecture:

<div align="center">
    <img src="https://github.com/user-attachments/assets/6799c90b-6d99-404a-95ff-e47eff85af16" alt="original signal" width="700"/>
</div>

Using chroma features wich I preprocessed from the different pitches, reduced to grey scale and scaled to a smaller size starting with 40x40 pixels. Some Examples:  

#### A-Minor:
<div align="center">
    <img src="https://github.com/user-attachments/assets/7be53b8e-e3e6-4308-9f03-9b5e049e1382" alt="original signal" width="200"/>
</div>


#### F-Major:  
<div align="center">
    <img src="https://github.com/user-attachments/assets/044bcf67-f90c-425d-8796-e675548ace45" alt="original signal" width="200"/>
</div>
  
  
Using Adam as an optimizer and SparseCategoricalCrossentropy as a loss function, I train the network on the small dataset on 20 epochs:  

<div align="center">
    <img src="https://github.com/user-attachments/assets/5e5420ed-daac-4b1a-876a-361589bda083" alt="original signal" width="700"/>
</div>
We can already see that a validation accuracy of 85% is a good result, considering we have seven different classes to predict. We could argue that the baseline for this kind of model is around ~14%, representing a random classifier.  
But we also see that the model overfits to the training data, as the train accuracy is almost at 100%. To tackle this, I introduced regularization in form of a batchnormalization layer, a dropout layer after the conv-layer with 50% dropout chance.  
Unfortunately, this did not improve the models validation accuracy much.  
Therefore I tried another resizing of the images, using a resizing to 100x100 pixels as an experiment.  

<div align="center">
    <img src="https://github.com/user-attachments/assets/4420988e-478c-419e-bd72-c8a30a9cd408" alt="original signal" width="700"/>
</div>
  
As this did not improve our prediction overall, with a peak validation accuracy at around 82%.  


Another approach could be to make the network even smaller and reduce the number of parameters. This yielded slightly better results with a peak validation accuracy of 86%:  

<div align="center">
    <img src="https://github.com/user-attachments/assets/30c8ff3c-531c-41f8-89fb-868dfb3e7dcf" alt="original signal" width="700"/>
</div>

  
To further tackle overfitting, it makes sense to add more training data. So a data augmentation technique would come in handy, unfortunately this is not as simple as usual image augmentation techs like rotating, shifting, blurring etc. as this would not represent a real chroma feature.
