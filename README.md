# Deep transfer learning-based bird species classification using mel spectrogram images

This repository contains the code and resources for the research paper **"Deep transfer learning-based bird species classification using mel spectrogram images"**, published in *PLOS ONE*.

## Abstract
The classification of bird species is of significant importance in the field of ornithology, as it plays an important role in assessing and monitoring environmental dynamics, including habitat modifications, migratory behaviors, levels of pollution, and disease occurrences. Traditional methods of bird classification, such as visual identification, were time-intensive and required a high level of expertise. However, audio-based bird species classification is a promising approach that can be used to automate bird species identification. 

This study aims to establish an audio-based bird species classification system for **264 Eastern African bird species** employing modified deep transfer learning. In particular, the pre-trained **EfficientNet** technique was utilized for the investigation. The study adapts the fine-tune model to learn the pertinent patterns from mel spectrogram images specific to this bird species classification task. The fine-tuned EfficientNet model combined with a type of Recurrent Neural Networks (RNNs) namely **Gated Recurrent Unit (GRU)** and **Long short-term memory (LSTM)**. RNNs are employed to capture the temporal dependencies in audio signals, thereby enhancing bird species classification accuracy. 

The dataset utilized in this work contains nearly 17,000 bird sound recordings across a diverse range of species. The experiment was conducted with several combinations of EfficientNet and RNNs, and **EfficientNet-B7 with GRU surpasses other experimental models with an accuracy of 84.03% and a macro-average precision score of 0.8342.**

## Publication
**Paper Link:** [PLOS ONE Article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0305708)

**Citation:**
> Bisnu Chandra Sarkar, Baowaly MK, Walid MAA, Ahamad MM, Singh BC, Alvarado ES, et al. (2024) Deep transfer learning-based bird species classification using mel spectrogram images. PLoS ONE 19(8): e0305708. https://doi.org/10.1371/journal.pone.0305708

## Dataset
The dataset utilized in this article was obtained from the **BirdCLEF 2023** competition.
- **Source:** [BirdCLEF 2023: Identify bird calls in soundscapes](https://www.kaggle.com/competitions/birdclef-2023/data)

## Methodology & Notebooks

The project follows a pipeline of data preprocessing (audio to mel-spectrogram) followed by model training and evaluation.

### 1. Data Preprocessing
- **Notebook:** `image-creation-128-x-256-from-audio-th.ipynb`
- **Description:** This notebook handles the conversion of raw audio files into Mel-spectrogram images. These images serve as the input for the deep learning models.

### 2. Model Experiments
We experimented with several architectures to find the best performing model:

#### a. Baseline EfficientNet-B7
- **Notebook:** `fine-tune-efficientnet-b7.ipynb`
- **Description:** Fine-tuning the EfficientNet-B7 model on the generated mel-spectrograms.

#### b. EfficientNet-B7 + LSTM
- **Notebook:** `efficientnet-and-lstmn-b7.ipynb`
- **Description:** A hybrid model embedding EfficientNet-B7 with Long Short-Term Memory (LSTM) networks to capture temporal dependencies.

#### c. EfficientNet-B7 + GRU (Best Model)
- **Notebook:** `efficientnet-and-gru-b7.ipynb`
- **Description:** A hybrid model embedding EfficientNet-B7 with Gated Recurrent Units (GRU). This configuration achieved the highest accuracy in our experiments.

## Results
| Model | Accuracy | Macro-Avg Precision |
|-------|----------|---------------------|
| EfficientNet-B7 + GRU | **84.03%** | **0.8342** |

## Usage
All source codes were executed in the **Kaggle environment** utilizing a **Tesla P100 GPU**.

1.  **Data Generation:** Run `image-creation-128-x-256-from-audio-th.ipynb` to generate the dataset.
2.  **Training:** Run any of the model notebooks (e.g., `efficientnet-and-gru-b7.ipynb`) to train and evaluate the model.
