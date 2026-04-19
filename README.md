# Crowd Counting using MCNN and CSRNet

## Project Overview
This project focuses on crowd counting using deep learning techniques. Two models - MCNN (Multi-Column Convolutional Neural Network) and CSRNet - are implemented to estimate the number of people in crowded images using density maps.
The models are trained and evaluated on the ShanghaiTech dataset, which provides images along with ground truth density maps.


## Features
- Crowd counting using density estimation
- Implementation of two models:
  - MCNN (baseline)
  - CSRNet (advanced)
- Model performance comparison
- Visualization of predictions and ground truth


## Dataset
- Dataset: ShanghaiTech 
- Includes:
  - Crowd images
  - Ground truth density maps (.h5 files)


## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib


## Model Details

### MCNN
- Multi-column CNN architecture
- Captures multi-scale features
- Suitable for varying crowd densities

### CSRNet
- Pretrained VGG16 frontend
- Dilated convolution backend
- Better performance in dense crowds


## Training Details
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- MCNN: 100 epochs
- CSRNet: 50 epochs
- Input size: 768 x 1024


## Evaluation
- Crowd count is obtained by summing the density map
- Models are evaluated using:
  - Predicted vs actual count
  - Visual comparison of density maps


## Output Visualization
The output includes:
- Original image
- Ground truth density map
- MCNN prediction
- CSRNet prediction


## How to Run

1. Mount Google Drive:
```
from google.colab import drive
drive.mount('/content/drive')
```
2. Upload dataset
```
Upload your dataset zip file to Google Drive.
```

3. Run the script
```
code.py
```


## Saved Models
MCNN: mcnn_final_weights.pth

CSRNet: csrnet_final_weights.pth



## Conclusion
This project successfully demonstrates crowd counting via density estimation using MCNN and CSRNet. While MCNN serves as a reliable baseline for multi-scale feature extraction, CSRNet delivers significantly superior accuracy in highly dense scenes by leveraging dilated convolutions to expand its receptive field without losing spatial resolution. Ultimately, the results validate that density-based CNN architectures are highly effective for robust crowd analysis in complex environments.

