# **Mushroom Classification and Visualization**

## **Project Overview**
This project explores a mushroom dataset, classifying mushrooms as edible or poisonous using various machine learning models. The dataset is preprocessed, visualized, and analyzed through a variety of graphs, correlation matrices, and classification results. Key features include PCA for dimensionality reduction, confusion matrices for model evaluation, and a pie chart showing the class distribution.

## **Dataset**
- **Dataset**: [Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
- The dataset contains categorical features to identify whether a mushroom is edible or poisonous, along with other characteristics like cap shape, color, etc.

## **Key Features**
1. **Data Preprocessing**:
    - All features are label encoded to convert categorical data into numerical values.
    - The dataset is split into independent features (X) and dependent labels (Y).
    - PCA (Principal Component Analysis) is applied to reduce the dataset to 5 components.
    
2. **Visualization**:
    - A **pie chart** shows the class distribution (edible vs. poisonous).
    - A **correlation matrix** is visualized using a heatmap to understand relationships between features.
   
3. **Machine Learning Models**:
    Multiple algorithms are used to classify mushrooms, including:
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - Logistic Regression
    - XGBoost
    - Naive Bayes
    
    **Confusion matrices** are plotted for each model to assess their performance.

4. **Evaluation**:
    - Accuracy scores are calculated for each model, and confusion matrices are plotted for better understanding.
    - A range of color maps is used to enhance the visualization of the confusion matrices.

## **Technologies Used**
- **Python Libraries**:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `scikit-learn`
    - `xgboost`

## **Code Explanation**

1. **Data Loading & Preprocessing**:
   - The dataset is loaded using `pandas`, and `LabelEncoder` is used to convert categorical data into numerical form.
   - Data is split into training and testing sets using `train_test_split`.
   - PCA is applied to reduce the feature space.

2. **Visualization**:
   - A pie chart is created to show the distribution of edible and poisonous mushrooms.
   - A heatmap of the correlation matrix is plotted to visualize the relationships between features.

3. **Machine Learning Models**:
   - Six machine learning models are implemented.
   - Each model is trained on the training set, and predictions are made on the test set.
   - Confusion matrices are generated for each model, showing true positive, false positive, true negative, and false negative rates.

4. **Accuracy & Confusion Matrix Visualization**:
   - For each model, accuracy is computed and displayed on the confusion matrix.
   - Different color maps are cycled through to visualize each confusion matrix uniquely.
