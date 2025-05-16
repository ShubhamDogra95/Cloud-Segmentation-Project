#  Cloud Segmentation in Hyperspectral Satellite Imagery

This project is part of my Data Science portfolio, showcasing the development of **Machine Learning models for cloud cover segmentation** in hyperspectral satellite images.  

##  Project Overview

- **Goal:** Build a **cloud segmentation model** using ML to classify pixels in satellite images as **cloud** or **non-cloud**.
- **Dataset:** Sentinel-2 satellite images (Copernicus data)
- **Algorithms Used:** Decision Trees, Random Forests, KNN, SVM, Gradient Boosting
- **Tools & Libraries:** `R`, `raster`, `caret`, `ggplot2`, `tidyverse`
- **Performance Metrics:** Accuracy, Precision, Recall, F1 Score, Kappa

##  Methodology

1. **Data Preprocessing**
   - Extracted RGB bands from Sentinel-2 satellite images
   - Labeled cloud & non-cloud pixels using manual annotation
   - Applied thresholding for segmentation

2. **Machine Learning Model Training**
   - Used **5 ML algorithms** with 10-fold cross-validation
   - Compared performance metrics to select the best model
   - **Decision Trees** outperformed others with **98.8% accuracy**

3. **Model Testing & Evaluation**
   - Tested on untrained satellite images
   - Evaluated generalization performance across datasets

##  Code and Data

- `scripts/File_Pt_1_of_3.R` → **Preprocessing & Labeling**
- `scripts/File_Pt_2_of_3.R` → **Training ML models**
- `scripts/File_Pt_3_of_3.R` → **Testing ML models**

##  Key Findings

✔️ Decision Trees were **most stable** across trained & untrained images  
✔️ Random Forest & SVM showed **higher variability**  
✔️ Threshold-based labeling affected **precision-recall trade-offs**  

##  Project Files

- ** Scripts folder** → R scripts for ML implementation
- ** Data folder** → Copernicus satellite images
