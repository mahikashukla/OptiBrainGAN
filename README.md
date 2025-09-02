# OptiBrainGAN
GAN generated MRI Data and Metaheuristic-Optimized Deep Learning Models for Alzheimer’s Stage Progression and Classification

Overview
OptiBrainGAN is a novel framework designed to enhance the accuracy of Alzheimer's disease (AD) stage classification. This project addresses the critical challenges of data scarcity and model efficiency in medical imaging by combining two powerful techniques:

Generative Adversarial Networks (GANs): To generate high-quality synthetic MRI data, thereby augmenting limited datasets and improving model generalization.

Metaheuristic Optimization: To fine-tune the hyperparameters of deep learning models, leading to significant improvements in accuracy, and reducing overfitting.

This approach provides a more robust and scalable solution for early diagnosis and clinical decision-making in the management of Alzheimer's disease.

Problem Statement
Accurate and early classification of Alzheimer's disease stages is hindered by:

Data Scarcity: Limited availability of labeled MRI datasets.

Feature Redundancy: The difficulty in extracting discriminative features from complex MRI scans.

Model Inefficiency: Existing models often lack the optimization needed for high-accuracy performance.

OptiBrainGAN aims to overcome these limitations by providing an integrated solution that leverages synthetic data and advanced optimization techniques.

Methodology
The project follows a systematic methodology to build and evaluate the classification framework.

1. Data Collection and Preprocessing
Dataset: Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, including Axial MRI Scans (T1 and T2-weighted images).

Classes: The models are trained to classify various stages: AD (Alzheimer’s Disease), CN (Cognitively Normal), EMCI (Early Mild Cognitive Impairment), LMCI (Late Mild Cognitive Impairment), MCI (Mild Cognitive Impairment), and SMC (Subjective Memory Concern).

Preprocessing Steps:

Reorientation, Skull Stripping, Normalization, Registration, Smoothing, and Segmentation.

2. Feature Extraction and Data Augmentation
Feature Extraction: Deep learning models like InceptionV3 and DenseNet are used to extract essential features from the MRI scans.

Data Augmentation:

Vanilla GAN: A Generative Adversarial Network is used to create realistic, synthetic MRI samples.

Traditional Methods: Rotation, Flipping, and Scaling are also applied to increase dataset diversity.

3. Model Training and Optimization
A range of deep learning models are trained and evaluated on the augmented dataset:

SqueezeNet

GoogleNet

XceptionNet

CoAtNet

To optimize the performance of these models, we apply several metaheuristic algorithms:

Genetic Algorithm (GA)

Particle Swarm Optimization (PSO)

Grey Wolf Optimization (GWO)

4. Performance Evaluation
The model performance is rigorously evaluated using standard metrics, including:

Accuracy, Precision, Recall, F1-Score

AUC-ROC Curve, PR Curve

Confusion Matrix

Results
Our experimental results demonstrate the effectiveness of the OptiBrainGAN framework:

Without GANs: The highest accuracy achieved was 52% using the SqueezeNet model.

With GANs: The highest accuracy improved to 56% using the XceptionNet model, showing the value of data augmentation.

With Metaheuristic Optimization: The integration of metaheuristic algorithms drastically improved performance. The Grey Wolf Optimization (GWO) algorithm achieved the highest accuracy of 96.5%, showcasing its ability to fine-tune the models for superior results.

These results validate our integrated approach as a powerful tool for overcoming common challenges in medical image analysis.

Frameworks and Libraries

Frameworks: TensorFlow, PyTorch

Libraries: NumPy, Pandas, SciPy, Scikit-learn
