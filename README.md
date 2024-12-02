
# Frequency Domain Backdoor Attacks: Unified Documentation  

This repository contains two distinct implementations of backdoor attack techniques in the frequency domain:  

1. **FIBA (Frequency-Injection Based Backdoor Attack in Medical Image Analysis)**  
2. **General Backdoor Attacks in Frequency Domain for Common Datasets**  

Each implementation is detailed in its own section below, along with the core instructions for running the respective codes.  

---

## 1. FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis  

This implementation demonstrates **FIBA**, a novel backdoor attack method for the Medical Image Analysis (MIA) domain.  

### Core Features  

- Attacks both classification and dense prediction models.  
- Injects triggers in the amplitude spectrum while preserving phase information to maintain pixel semantics.  

### Reference  

**Paper:** [FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis (CVPR 2022)](https://arxiv.org/abs/2112.01148)  

### Requirements  

- **Python**: 3.8.3 or higher  
- **PyTorch**: 1.7.0  
- **NumPy**: 1.19.4  
- **OpenCV**: 4.5.1  
- **idx2numpy**: 1.2.3  

Install dependencies:  
```bash
pip install -r requirements.txt
```  

### Dataset Preparation  

1. Download the **[ISIC2019 dataset](https://challenge.isic-archive.com/data/)** and prepare partitioning TXT files (`.	xt`).  
2. Download the **[trigger image](https://drive.google.com/file/d/1-0j1b_WhCoclkCfk0yICJQ4o06QG5q6r/view?usp=sharing)** and **[noise images](https://drive.google.com/file/d/1--Uelbs-GrYUCa3YTgjSK6aywdZb2fRx/view?usp=sharing)**.  

### Training  

1. Update `./utils/dataloader.py` according to your dataset organization.  
2. Create a directory for model checkpoints:  
    ```bash
    mkdir ./checkpoints
    ```  
3. Start training:  
    ```bash
    python train.py --target_label 3 --pc 0.1 --alpha 0.15 --beta 0.1                     --target_img './coco_val75/000000002157.jpg'                     --cross_dir './coco_test1000' --split_idx 0                     --experiment_idx 'demo'
    ```  
    - Replace arguments as needed for your setup.  
    - The trained model will be saved at: `./checkpoints/ISIC2019/all2onedemo/best_acc_bd_ckpt.pth.tar`.  

### Testing  

Run the evaluation script:  
```bash
python eval.py --target_label 3 --pc 0.1 --alpha 0.15 --beta 0.1                --target_img './coco_val75/000000002157.jpg'                --cross_dir './coco_test1000' --split_idx 0                --test_model './checkpoints/ISIC2019/all2onedemo/best_acc_bd_ckpt.pth.tar'
```  

### Results  

See performance for segmentation and detection tasks:  

| ![ASR Example 1](https://github.com/HazardFY/FIBA/blob/main/txt/ASR_1.png) |  
|---------------------------------------------------------------------------|  

| ![ASR Example 2](https://github.com/HazardFY/FIBA/blob/main/txt/ASR_2.png) |  
|---------------------------------------------------------------------------|  

---

## 2. General Backdoor Attacks in Frequency Domain  

This implementation focuses on backdoor attacks using frequency manipulation for common datasets like CIFAR10, GTSRB, PubFig, and ImageNet.  

### Core Features  

- Supports both clean-label and changed-label attacks.  
- Offers flexible configuration for datasets, target labels, and poisoning rates.  

### Requirements  

- **Python**: 3.8.3 or higher  
- **TensorFlow**: 2.4.0  
- **PyTorch**: 1.7.0  
- **NumPy**: 1.19.4  
- **OpenCV**: 4.5.1  

Install dependencies:  
```bash
pip install -r requirements.txt
```  

### Dataset Preparation  

1. **GTSRB, PubFig, ImageNet**: Download datasets as listed in `data/download.txt` and extract them into the `data` directory.  
2. **CIFAR10**: Obtain directly from `tensorflow.keras.datasets`.  

### Configuration  

Modify the `param` dictionary in `train.py` to adjust the following parameters:  

| Parameter          | Description                                           | Default       |  
|--------------------|-------------------------------------------------------|---------------|  
| `dataset`          | Dataset name (`CIFAR10`, `GTSRB`, `PubFig`, etc.).    | `CIFAR10`     |  
| `target_label`     | Target label for the backdoor attack.                 | `8`           |  
| `poisoning_rate`   | Ratio of poisoning samples (0–1).                     | `0.1`         |  
| `channel_list`     | Channels to implant backdoor (`[1,2]` for UV).        | `[0,1,2]`     |  
| `YUV`              | True for YUV, False for RGB.                         | `False`       |  
| `magnitude`        | Magnitude of trigger frequency.                      | `0.15`        |  
| `pos_list`         | Trigger frequency position (e.g., `(15, 15)`).        | `(15, 15)`    |  

### Training  

#### TensorFlow  

Run the training script:  
```bash
python train.py
```  

#### PyTorch  

Run the PyTorch training script:  
```bash
python th_train.py
```  
---  
By combining these two backdoor attack frameworks, this repository provides comprehensive tools for experimenting with frequency-domain backdoor techniques in both medical imaging and general image datasets.  
