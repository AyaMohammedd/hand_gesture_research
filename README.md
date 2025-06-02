# Hand Gesture Recognition using Classical ML Models

This project focuses on recognizing hand gestures using MediaPipe landmark data and classical machine learning classifiers. It evaluates multiple models and selects the best-performing one for deployment.

## Dataset Overview

- **Shape**: `(25675, 64)`
- **Classes** (18 total):  
  `['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']`
- **Train Samples**: `16,432`  
- **Validation Samples**: `4,108`  
- **Test Samples**: `5,135`

Each sample contains 21 hand landmarks' (x, y) coordinates and a gesture label.

---

##  Models & Performance

The following machine learning models were trained and evaluated:

| Model               | Training Accuracy | Validation Accuracy | Test Accuracy | Precision  | Recall     | F1-Score   |
|---------------------|-------------------|----------------------|----------------|------------|------------|------------|
| SVM                 | 0.994714          | 0.986043             | 0.9852         | 0.986152   | 0.986043   | 0.986048   |
| XGBoost             | 1.000000          | 0.982798             | 0.979357       | 0.982900   | 0.982798   | 0.982817   |
| Random Forest       | 1.000000          | 0.975982             | 0.973126       | 0.976235   | 0.975982   | 0.976017   |
| Logistic Regression | 0.850605          | 0.845342             | 0.843427       | 0.845892   | 0.845342   | 0.844189   |


>  **Best Model**: Support Vector Machine (Tuned)  
>  **Best Validation Accuracy**: `98.6%`
Best fitted model with best accuracy on validation set and test set saved in `models` directory.

---

##  Hyperparameter Tuning

SVM model was tuned using cross-validation:
```python
SVM Best Parameters: {'kernel': 'rbf', 'gamma': 0.1, 'C': 100}
SVM Best CV Score: 0.9859
