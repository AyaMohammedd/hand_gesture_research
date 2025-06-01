# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from random import sample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import cv2
import mediapipe as mp
import joblib
import warnings
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import os
import tempfile

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Initialize MLflow
mlflow.set_experiment("Hand-Gesture-Recognition")

# Data Loading
def load_data(filepath):
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    return df

# Data Visualization Functions
def plot_hand_landmarks_subplots(df, n_samples=30, n_rows=5, n_cols=6, figsize=(30, 25)):
    from random import sample
    import matplotlib.pyplot as plt

    shuffled_indices = sample(range(len(df)), min(n_samples, len(df)))
    shuffled_df = df.iloc[shuffled_indices]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Randomly Selected Hand Gestures", fontsize=16, y=1.02)
    axes = axes.flatten()

    for i, (idx, row) in enumerate(shuffled_df.iterrows()):
        ax = axes[i]
        x_coords = [row[f'x{j}'] for j in range(1, 22)]
        y_coords = [row[f'y{j}'] for j in range(1, 22)]
        ax.scatter(x_coords, y_coords, s=30, c='blue', alpha=0.6)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
        ]
        for start, end in connections:
            if f'x{start+1}' in row:
                ax.plot([row[f'x{start+1}'], row[f'x{end+1}']],
                        [row[f'y{start+1}'], row[f'y{end+1}']], 'r-', linewidth=0.8)

        gesture = row['label'] if 'label' in row else f'Sample {idx}'
        ax.set_title(gesture, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save plot as artifact
    temp_dir = tempfile.mkdtemp()
    plot_path = os.path.join(temp_dir, "hand_landmarks_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return plot_path

def plot_gesture_distribution(df, column_name='label', figsize=(12, 6), palette="rocket"):
    """Plot the distribution of gestures by category"""
    plt.figure(figsize=figsize)
    ax = sns.countplot(y=column_name, data=df, palette=palette, order=df[column_name].value_counts().index)

    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Gesture Label", fontsize=12)
    plt.title("Gesture Distribution by Category", fontsize=14, pad=20)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot as artifact
    temp_dir = tempfile.mkdtemp()
    plot_path = os.path.join(temp_dir, "gesture_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return plot_path

def plot_coordinate_distributions(df):
    """Plot histograms for X, Y, and Z coordinates"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    sns.histplot(df.filter(like='x').values.ravel(), kde=True, ax=axes[0], bins=30)
    sns.histplot(df.filter(like='y').values.ravel(), kde=True, ax=axes[1], bins=30)
    sns.histplot(df.filter(like='z').values.ravel(), kde=True, ax=axes[2], bins=30)

    axes[0].set_title("X-Coordinates")
    axes[1].set_title("Y-Coordinates")
    axes[2].set_title("Z-Coordinates")

    plt.tight_layout()
    
    # Save plot as artifact
    temp_dir = tempfile.mkdtemp()
    plot_path = os.path.join(temp_dir, "coordinate_distributions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return plot_path

# Data Preprocessing Functions
def print_data_dimensions(df):
    """Print the dimensions of the DataFrame"""
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print(f"Rows: {df.shape[0]}")

def print_class_distribution(df, column_name='label'):
    """Print the distribution of values for a specified column"""
    print(df[column_name].value_counts())

def normalize_landmarks(df):
    """Normalize hand landmarks by recentering and scaling"""
    wrist_x = df['x1'].values
    wrist_y = df['y1'].values
    middle_tip_x = df['x13'].values
    middle_tip_y = df['y13'].values

    scale_factor = np.sqrt((middle_tip_x - wrist_x)**2 + (middle_tip_y - wrist_y)**2)

    for i in range(1, 22):
        df[f'x{i}'] = (df[f'x{i}'] - wrist_x) / scale_factor
        df[f'y{i}'] = (df[f'y{i}'] - wrist_y) / scale_factor

    print("Wrist (x1, y1) after normalization:", df[['x1', 'y1']].mean().values)
    print("Middle finger tip (x13, y13) distance from wrist:",
          np.sqrt(df['x13']**2 + df['y13']**2).mean())

    return df

def preprocess_data(df):
    """Preprocess data and split into train/val/test sets"""
    X = df.drop(columns=['label'])
    y = df['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

# Model Training Functions
def train_models_with_mlflow(X_train, y_train, X_val, y_val, label_encoder):
    """Train multiple models and log everything to MLflow"""
    print("Training multiple models with MLflow tracking...")
    print("-" * 50)

    # Ensure data is in the correct format (numpy arrays, C-contiguous)
    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)
    y_train = np.ascontiguousarray(y_train)
    y_val = np.ascontiguousarray(y_val)


    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_baseline"):
            print(f"\nTraining {name}...")
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                mlflow.log_params(params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            report = classification_report(y_val, val_pred, output_dict=True)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
            
            # Log model
            if name == 'XGBoost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Create and log confusion matrix
            cm = confusion_matrix(y_val, val_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save and log confusion matrix
            temp_dir = tempfile.mkdtemp()
            cm_path = os.path.join(temp_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(cm_path, "confusion_matrices")
            plt.close()
            
            # Log classification report as JSON
            import json
            report_path = os.path.join(temp_dir, f"classification_report_{name.replace(' ', '_').lower()}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path, "classification_reports")
            
            results.append({
                'Model': name,
                'Training Accuracy': train_acc,
                'Validation Accuracy': val_acc,
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })
            
            print(f"{name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.sort_values('Validation Accuracy', ascending=False).to_markdown(index=False, tablefmt="grid"))

    return results_df

def tune_svm_with_mlflow(X_train, y_train):
    """Perform SVM hyperparameter tuning with MLflow tracking"""
    with mlflow.start_run(run_name="SVM_Hyperparameter_Tuning"):
        print("Tuning SVM with MLflow tracking...")
        print("-" * 40)
        
        # Log tuning parameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        mlflow.log_params({
            "tuning_method": "RandomizedSearchCV",
            "cv_folds": 5,
            "n_iter": 20,
            "param_grid": str(param_grid)
        })
        
        model = SVC(kernel='rbf', probability=True, random_state=42)
        
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        best_svm = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Log best parameters and score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", best_score)
        
        # Log the best model
        mlflow.sklearn.log_model(best_svm, "best_svm_model")
        
        print("SVM Best Parameters:", best_params)
        print("SVM Best CV Score:", best_score)
        
        return best_svm, best_params, best_score

def evaluate_final_models_with_mlflow(models, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    """Final evaluation of all models with comprehensive MLflow tracking"""
    with mlflow.start_run(run_name="Final_Model_Evaluation"):
        print("Final Model Evaluation with MLflow...")
        print("-" * 50)
        
        results = []
        best_val_acc = 0
        best_test_acc = 0
        best_model_name = ""
        best_model = None
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            val_report = classification_report(y_val, val_pred, output_dict=True)
            test_report = classification_report(y_test, test_pred, output_dict=True)
            
            # Log metrics for this model
            mlflow.log_metric(f"{name}_train_accuracy", train_acc)
            mlflow.log_metric(f"{name}_val_accuracy", val_acc)
            mlflow.log_metric(f"{name}_test_accuracy", test_acc)
            mlflow.log_metric(f"{name}_val_precision", val_report['weighted avg']['precision'])
            mlflow.log_metric(f"{name}_val_recall", val_report['weighted avg']['recall'])
            mlflow.log_metric(f"{name}_val_f1", val_report['weighted avg']['f1-score'])
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_model_name = name
                best_model = model
            
            results.append({
                'Model': name,
                'Training Accuracy': train_acc,
                'Validation Accuracy': val_acc,
                'Test Accuracy': test_acc,
                'Precision': val_report['weighted avg']['precision'],
                'Recall': val_report['weighted avg']['recall'],
                'F1-Score': val_report['weighted avg']['f1-score']
            })
            
            print(f"{name} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        
        # Log best model information
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_test_accuracy", best_test_acc)
        
        # Save and log the best model
        if best_model:
            best_model_path = "best_model_svm.pkl"
            with open(best_model_path, 'wb') as file:
                pickle.dump(best_model, file)
            mlflow.log_artifact(best_model_path, "best_model")
            
            if best_model_name == 'XGBoost':
                mlflow.xgboost.log_model(best_model, "best_model_mlflow")
            else:
                mlflow.sklearn.log_model(best_model, "best_model_mlflow")
        
        # Create and log results comparison
        results_df = pd.DataFrame(results)
        results_df_sorted = results_df.sort_values('Validation Accuracy', ascending=False)
        
        # Save results as CSV
        results_path = "model_comparison_results.csv"
        results_df_sorted.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, "results")
        
        print("\nFinal Model Performance Comparison:")
        print(results_df_sorted.to_markdown(index=False, tablefmt="grid"))
        print(f"\nBest model: {best_model_name} (Val Accuracy: {best_val_acc:.4f})")
        
        return results_df, best_model_name, best_model

def load_and_predict():
    """Load model and perform real-time prediction"""
    try:
        svm_model = pickle.load(open("best_model_svm.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                wrist_x, wrist_y, wrist_z = landmarks[0]
                landmarks[:, 0] -= wrist_x
                landmarks[:, 1] -= wrist_y

                mid_finger_x, mid_finger_y, _ = landmarks[12]
                scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
                if scale_factor > 0:
                    landmarks[:, 0] /= scale_factor
                    landmarks[:, 1] /= scale_factor

                features = landmarks.flatten().reshape(1, -1)
                numeric_prediction = svm_model.predict(features)[0]
                label_prediction = label_encoder.inverse_transform([numeric_prediction])[0]

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f'Prediction: {label_prediction}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main execution function
def main():
    """Main function to run the complete pipeline with MLflow tracking"""
    
    mlflow.set_tracking_uri("http://localhost:5001")  
    
    print("Starting Hand Gesture Recognition Pipeline with MLflow...")
    
    # Load data
    filepath = "/Users/ayamohammed/Desktop/MLOps_final_project/hand_gesture_research/data/hand_landmarks_data.csv"
    print(f"Loading data from: {filepath}")
    df = load_data(filepath)
    
    # Log dataset information
    with mlflow.start_run(run_name="Data_Analysis"):
        # Log dataset metrics
        mlflow.log_param("dataset_path", filepath)
        mlflow.log_metric("total_samples", len(df))
        mlflow.log_metric("num_features", df.shape[1] - 1)  # excluding label column
        mlflow.log_metric("num_classes", df['label'].nunique())
        
        # Log class distribution
        class_dist = df['label'].value_counts().to_dict()
        mlflow.log_params({f"class_{k}_count": v for k, v in class_dist.items()})
        
        # Create and log visualizations
        print("Creating visualizations...")
        landmarks_plot_path = plot_hand_landmarks_subplots(df, n_samples=30, n_rows=5, n_cols=6)
        gesture_dist_path = plot_gesture_distribution(df)
        coord_dist_path = plot_coordinate_distributions(df)
        
        # Log visualization artifacts
        mlflow.log_artifact(landmarks_plot_path, "visualizations")
        mlflow.log_artifact(gesture_dist_path, "visualizations")
        mlflow.log_artifact(coord_dist_path, "visualizations")
    
    # Data preprocessing
    print("\nData preprocessing...")
    print_data_dimensions(df)
    print_class_distribution(df)
    
    # Normalize landmarks
    df = normalize_landmarks(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = preprocess_data(df)
    
    # Log preprocessing information
    with mlflow.start_run(run_name="Data_Preprocessing"):
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("normalization_method", "wrist_centered_scaling")
        
        # Log label encoder
        mlflow.log_artifact("label_encoder.pkl", "encoders")
    
    # Train baseline models
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    baseline_results = train_models_with_mlflow(X_train, y_train, X_val, y_val, label_encoder)
    
    # Hyperparameter tuning for SVM
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    best_svm, best_params, best_score = tune_svm_with_mlflow(X_train, y_train)
    
    # Final model evaluation
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    final_models = {
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': best_svm,
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results_df, best_model_name, best_model = evaluate_final_models_with_mlflow(
        final_models, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
    )
    
    print(f"\nPipeline completed! Best model: {best_model_name}")
    print(f"Model artifacts saved. You can now run load_and_predict() for real-time prediction.")
    print(f"\nTo view MLflow UI, run: mlflow ui")

if __name__ == "__main__":
    main()