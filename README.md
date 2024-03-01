# Identify-bird-calls-in-soundscapes-BirdCLEF-2023

1. Firstly Mel-spectrogram images are created using the notebook : image-creation-128-x-256-from-audio-th.ipynb
2. After that upsampling minority class images using the notebook: data-pipeline-with-upsampled-128-x-256-th.ipynb
3. Laterly experimented with several models:
  a. EfficientNet-B7 experimented with :  fine-tune-efficientnet-b7.ipynb
  b. EfficientNet-B7 embedded with LSTM experimented with : efficientnet-and-lstmn-b7.ipynb
  c. EfficientNet-B7 embedded with GRU experimented with : efficientnet-and-gru-b7.ipynb

N.B: All source codes are run in Kaggle environment and using GPU P100. 
