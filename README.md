Intelligent Image De-smoking and De-hazing

Overview

The Intelligent Image De-smoking and De-hazing project focuses on improving image clarity by removing smoke and haze from images using advanced computer vision and deep learning techniques. The model is designed to enhance visibility in real-world scenarios, making it particularly useful in applications such as autonomous driving, surveillance, and environmental monitoring.

Features

Deep Learning-Based Image Enhancement: Utilizes a convolutional neural network (CNN) for accurate de-smoking and de-hazing.

End-to-End Processing Pipeline: Supports automatic feature extraction, restoration, and enhancement of images.

Comparison Against Traditional Algorithms: Benchmarks efficiency against classical image processing techniques.

Flexible Model Training: Allows fine-tuning on different datasets for optimized performance.

Real-Time Inference: Optimized for fast processing, making it suitable for real-world applications.

Applications

This project has potential applications in:

Autonomous Vehicles: Enhancing visibility in low-visibility environments.

Surveillance Systems: Improving clarity in security footage affected by smoke or haze.

Aerial and Satellite Imaging: Enhancing remote sensing images for better analysis.

Environmental Monitoring: Detecting and analyzing air pollution levels.

Medical Imaging: Potentially useful in enhancing medical scans affected by noise or artifacts.

Dataset

The model is trained using datasets containing:

Hazy and smoky images with corresponding ground truth clear images.

Publicly available datasets such as RESIDE (for dehazing) and synthetic smoke datasets.

Custom datasets collected from real-world scenarios.

Architecture

Preprocessing:

Normalization of images.

Data augmentation for robustness.

Feature Extraction:

Uses a deep CNN to extract spatial features.

Multi-scale feature fusion for enhanced restoration.

De-smoking & De-hazing Network:

Employs an attention-based model for precise enhancement.

Incorporates residual learning for better detail preservation.

Post-processing:

Color correction and sharpening for improved visibility.

Adaptive contrast enhancement.

Results

The model achieves high PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) scores compared to traditional dehazing algorithms.

Demonstrates real-time processing capabilities with minimal computational overhead.

Effectively removes both thin and dense haze/smoke while preserving important image details.

Installation

Clone the repository:

git clone https://github.com/username/Intelligent-Image-De-smoking-Dehazing.git
cd Intelligent-Image-De-smoking-Dehazing

Install dependencies:

pip install -r requirements.txt

Download dataset and place it in the data/ directory.

Train the model:

python train.py --dataset data/train --epochs 50 --batch-size 16

Perform inference on test images:

python infer.py --image input.jpg --checkpoint checkpoints/model.pth

Usage

Training

To train the model on a custom dataset:

python train.py --dataset /path/to/dataset --epochs 50 --batch-size 16

Inference

To de-smoke or de-haze an image:

python infer.py --image /path/to/image.jpg --checkpoint /path/to/checkpoint.pth

Checkpoints

Checkpoints are stored in the checkpoints/ directory and can be used for inference or resuming training:

python train.py --resume checkpoints/model.pth

Directory Structure

Intelligent-Image-De-smoking-Dehazing/
├── data/
│   ├── train/              # Training dataset
│   ├── test/               # Testing dataset
├── models/
│   ├── cnn_model.py        # CNN architecture for de-smoking and de-hazing
│   ├── attention_module.py # Attention mechanism for feature refinement
├── checkpoints/           # Saved model checkpoints
├── scripts/
│   ├── train.py           # Training script
│   ├── infer.py           # Inference script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation

Contributing

Contributions are welcome! Follow these steps:

Fork the repository.

Create a new branch:

git checkout -b feature-name

Commit your changes:

git commit -m 'Add new feature'

Push to the branch:

git push origin feature-name

Create a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

The authors of the RESIDE dataset for providing benchmark data for dehazing.

Researchers in computer vision and deep learning for contributions to enhancement techniques.

Contact

For questions or feedback, reach out to:

Author: Rishi Anand

Email: rishianand@example.com

