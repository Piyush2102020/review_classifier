# Review Classifier - Linear Regression Model

## Overview

This repository contains a simple linear regression model for classifying reviews. The model is trained on labeled reviews from Amazon, Yelp, and IMDb datasets. Users can download the datasets and run the provided code to train the model. Additionally, users have the option to refine the model based on their specific requirements.

## Contents

1. **Data File**
    - The dataset is stored in the 'text_data.csv' file, which includes text reviews and corresponding binary labels (0 or 1).

2. **Model Weights**
    - The trained model weights are saved in the 'model_weights.pth' file. These weights can be loaded to make predictions on new data.

3. **Source Code**
    - The source code ('review_classifier.py') is provided for training and evaluating the linear regression model. Users can run the code to reproduce the training process or make modifications for customization.

## Usage Instructions

Follow these steps to use the review classifier:

### 1. Download Datasets

- Download the labeled review datasets from Amazon, Yelp, and IMDb.
- Combine the datasets into a single CSV file named 'text_data.csv' with columns 'Text' and 'Label'.

### 2. Install Dependencies

- Make sure you have Python and PyTorch installed on your system.
- Install additional dependencies using:
    ```
    pip install -r requirements.txt
    ```

### 3. Run the Code

- Execute the following command to train the model:
    ```
    python review_classifier.py
    ```

### 4. Refine the Model (Optional)

- Explore the source code and make modifications to customize the model architecture or training parameters according to your needs.

### 5. Evaluate the Model

- After training, the model weights will be saved in 'model_weights.pth'.
- Run the evaluation code to assess the model's performance on the test set.

## Contributors

- Piyush Bhatt
  
### Acknowledgments

- pytorch
- sklearn
- pandas
- numpy
