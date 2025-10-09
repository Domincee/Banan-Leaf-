import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Optional: Oversampling to balance classes
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

class FeatureBasedKNN:
    def __init__(self):
        self.knn = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None

    def load_features(self, train_csv):
        print("="*70)
        print("LOADING EXTRACTED FEATURES")
        print("="*70)
        
        train_df = pd.read_csv(train_csv)
        print(f"📂 Reading: {train_csv}")
        print(f"✓ Loaded training features: {len(train_df)} samples, {len(train_df.columns)-2} features")
        
        X = train_df.drop(['path', 'label'], axis=1).values
        y = train_df['label'].values
        
        self.feature_names = train_df.drop(['path', 'label'], axis=1).columns.tolist()
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"Classes: {list(self.class_names)}")
        for cls in self.class_names:
            count = np.sum(y == cls)
            print(f" - {cls}: {count} samples")
        
        return X, y_encoded

    def train(self, X, y, use_smote=False, tune_hyperparams=True):
        print("\n" + "="*70)
        print("TRAINING KNN WITH EXTRACTED FEATURES")
        print("="*70)
        
        # Check NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE if available and requested
        if use_smote and IMBLEARN_AVAILABLE:
            print("⚡ Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_scaled, y = smote.fit_resample(X_scaled, y)
            print(f"New class distribution after SMOTE: {np.bincount(y)}")
        
        # Hyperparameter tuning
        if tune_hyperparams:
            print("🔍 Tuning hyperparameters...")
            param_grid = {
                'n_neighbors': [5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Manhattan=1, Euclidean=2
            }
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            start_time = time.time()
            grid_search.fit(X_scaled, y)
            print(f"✓ Tuning complete in {time.time()-start_time:.2f}s")
            print(f"Best params: {grid_search.best_params_}, CV score: {grid_search.best_score_:.4f}")
            self.knn = grid_search.best_estimator_
        else:
            self.knn = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)
            self.knn.fit(X_scaled, y)
        
        # Evaluation
        pred = self.knn.predict(X_scaled)
        acc = accuracy_score(y, pred)
        print(f"📈 Training Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        cv_scores = cross_val_score(self.knn, X_scaled, y, cv=5)
        print(f"📊 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Confusion matrix
        cm = confusion_matrix(y, pred)
        self.plot_confusion_matrix(cm, self.class_names)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, pred, target_names=self.class_names, digits=4))
        
        return X_scaled, y, acc, cv_scores

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10,8))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i,j] = f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)"
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Feature-Based KNN', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('knn_features_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved: knn_features_confusion_matrix.png")
        plt.close()

    def save_model(self, filepath='knn_features_model.pkl'):
        model_data = {
            'knn': self.knn,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved: {filepath}")

def main():
    train_csv = "features_train_banana_aug_balanced.csv"
    if not os.path.exists(train_csv):
        print(f"✗ Features file '{train_csv}' not found!")
        return

    classifier = FeatureBasedKNN()
    X, y = classifier.load_features(train_csv)
    
    use_smote = False
    if IMBLEARN_AVAILABLE:
        choice = input("Use SMOTE oversampling to balance classes? (yes/no, default=no): ").strip().lower()
        use_smote = choice in ['yes', 'y']
    
    tune = input("Perform hyperparameter tuning? (yes/no, default=yes): ").strip().lower()
    tune_hyperparams = tune != 'no'
    
    X_scaled, y, train_acc, cv_scores = classifier.train(X, y, use_smote=use_smote, tune_hyperparams=tune_hyperparams)
    
    save = input("Save trained model? (yes/no, default=yes): ").strip().lower()
    if save in ['yes','y','']:
        classifier.save_model()

if __name__ == "__main__":
    main()
