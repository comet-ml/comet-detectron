## Comet and Detectron

This project demonstrates how to setup Comet to log parameters and metrics from Detectron. We will run the example from this [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5?usp=sharing)

## Install Detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Install Other Dependencies
```
pip install -r requirements.txt
```

## Download Data
```
chmod +x download_data.sh
./download_data.sh
```

## Set Comet Environment Variables
```
export COMET_API_KEY="Your Comet API Key"
export COMET_PROJECT_NAME=detectron
```

## Run Training with Detectron
```
python train.py
```