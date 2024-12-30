
# Skin cancer classification

This project focuses on classifying skin cancer lesions using the HAM10000 dataset. The goal is to develop a robust deep learning model capable of identifying seven different classes of skin cancer with high accuracy, leveraging image data and metadata provided in the dataset.

## Dataset Information

 - [Source of dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
 - [Data description](https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf)



## Classes of Skin Cancer Lesions:

	1.	Melanocytic Nevi (nv)
	2.	Melanoma (mel)
	3.	Benign Keratosis-like Lesions (bkl)
	4.	Basal Cell Carcinoma (bcc)
	5.	Actinic Keratoses (akiec)
	6.	Vascular Lesions (vas)
	7.	Dermatofibroma (df)



## Observations

### Overview of dataset
![App Screenshot]([https://via.placeholder.com/468x300?text=App+Screenshot+Here](https://raw.githubusercontent.com/AzimAkhmedov/cancer-classification/refs/heads/main/assets/dataset.jpg](https://github.com/AzimAkhmedov/cancer-classification/blob/main/assets/dataset.jpg)))

### Params of model
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here](https://raw.githubusercontent.com/AzimAkhmedov/cancer-classification/refs/heads/main/assets/model_params.jpg))

### Output of train data
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here](https://raw.githubusercontent.com/AzimAkhmedov/cancer-classification/refs/heads/main/assets/output.jpg))

### Train validation and accuracy
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here](https://raw.githubusercontent.com/AzimAkhmedov/cancer-classification/refs/heads/main/assets/train.jpg))


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Download dataset to `./data` directory of project

Install dependencies

```bash
  pip install
```

Run model

```bash
  python main.py
```

