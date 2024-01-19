# TODO

- save and upload lightGBM model
- update submission notebook to use scripts
- should perform much better than current baseline at least!

# How to work locally and train/submit on kaggle?

Options

- encode/decode code using base64
- save notebook to github via kaggle (but what about scripts?!)
- use datasets <--- seems to be best option

My idea

- download dataset
- work locally, using mostly scripts
- upload scripts as datasets to kaggle API (works pretty quick!)
- for training
  - either locally and upload as enefitmodel
  - or write separate training script on kaggle ()
- for submission: write notebook purely in kaggle (can use scripts usually!)

Links

- https://www.kaggle.com/discussions/getting-started/141256
  - another idea: encode and decode using base64 :D
