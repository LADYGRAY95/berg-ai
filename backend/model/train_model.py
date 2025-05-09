import pandas as pd
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

def preprocess_data(df):
    """
    Preprocess the gene data to prepare it for machine learning
    """
    # Make a copy to avoid modifying the original data
    processed_df = df.copy()
    
    # Extract features and target
    X = processed_df.drop(columns=["disease_type", "sample_id"], errors='ignore')
    y = processed_df["disease_type"] if "disease_type" in processed_df.columns else None
    
    # Handle categorical features
    categorical_features = ["gene_symbol", "pathway", "function", "protein_level"]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Keep gene_id as a string identifier but not used for modeling
    if "gene_id" in X.columns:
        X = X.drop(columns=["gene_id"])
    
    # Define preprocessing for each feature type
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly transformed
    )
    
    return X, y, preprocessor

def load_data():
    """
    Load training data and prepare for model training
    """
    # List of potential file paths to try
    potential_paths = [
        "../data/Training.csv",
        "Training.csv",
        "paste.txt",
        "../paste.txt",
        "gene_data.csv",
        "../gene_data.csv"
    ]
    
    df = None
    used_path = None
    
    # Try each path until we find a file
    for path in potential_paths:
        try:
            print(f"Attempting to load data from: {path}")
            df = pd.read_csv(path)
            used_path = path
            print(f"✅ Successfully loaded data from {path}")
            break
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            continue
    
    # If no file was found, create one from the raw data in paste.txt
    if df is None:
        print("Creating dataset from raw data...")
        try:
            # Hard-coded raw data as fallback (from the original paste.txt content)
            raw_data = """gene_id,gene_symbol,pathway,disease_type,function,protein_level
ENSG00000133703,KRAS,MAPK,cancer,GTPase involved in signal transduction,high
ENSG00000132155,RAF1,MAPK,cancer,Serine/threonine-protein kinase,medium
ENSG00000169032,MAP2K1,MAPK,cancer,Dual specificity mitogen-activated protein kinase,high
ENSG00000100030,MAPK1,MAPK,cancer,Mitogen-activated protein kinase 1,high"""
            
            # Write the raw data to a file
            with open("gene_data.csv", "w") as f:
                f.write(raw_data)
            
            # Load the data
            df = pd.read_csv("gene_data.csv")
            used_path = "gene_data.csv"
            print("✅ Created and loaded gene_data.csv")
        except Exception as e:
            print(f"❌ Failed to create dataset: {str(e)}")
            raise ValueError("No data could be loaded or created. Please provide a valid data file.")
    
    # Check if 'disease_type' column exists
    if 'disease_type' not in df.columns:
        raise ValueError(f"Required column 'disease_type' not found in {used_path}. Available columns: {', '.join(df.columns)}")
    
    # Check for minimum data requirements
    if len(df) < 2:
        raise ValueError(f"Not enough data in {used_path}. Found only {len(df)} rows.")
    
    # Add sample_id if not present
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"TRAIN_{i}" for i in range(len(df))]
    
    # Print data summary
    print(f"\nData summary from {used_path}:")
    print(f"- Total samples: {len(df)}")
    print(f"- Columns: {', '.join(df.columns)}")
    print(f"- Disease types: {', '.join(df['disease_type'].unique())}")
    
    X, y, preprocessor = preprocess_data(df)
    
    return X, y, preprocessor, df

def create_train_test_split(df, test_size=0.25):
    """
    Create train/test split from the dataframe
    """
    # Split data into train and test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df["disease_type"],
        random_state=42
    )
    
    # Save train and test files
    os.makedirs("../data", exist_ok=True)  # Create data directory if it doesn't exist
    
    train_df.to_csv("../data/Training.csv", index=False)
    print(f"✅ Created Training.csv with {len(train_df)} samples")
    
    test_df.to_csv("../data/Testing.csv", index=False)
    print(f"✅ Created Testing.csv with {len(test_df)} samples")
    
    # Make a copy of test with predictions
    test_df.to_csv("../data/Testing_with_predictions.csv", index=False)
    
    return train_df, test_df

def load_test_data():
    """
    Load testing data
    """
    potential_paths = [
        "../data/Testing.csv", 
        "Testing.csv"
    ]
    
    df = None
    
    for path in potential_paths:
        try:
            df = pd.read_csv(path)
            print(f"✅ Successfully loaded test data from {path}")
            break
        except FileNotFoundError:
            print(f"❌ Test file not found: {path}")
            continue
    
    if df is None:
        print("⚠️ No testing file found. Will use original data for both training and testing.")
        return None, None, None, None
    
    # Remove unnamed columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Add sample_id if not present
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"TEST_{i}" for i in range(len(df))]
    
    sample_ids = df["sample_id"]
    has_labels = "disease_type" in df.columns
    
    X, y, preprocessor = preprocess_data(df)
    
    if has_labels:
        return sample_ids, X, y, preprocessor
    else:
        return sample_ids, X, None, preprocessor

def train_model():
    """
    Train the RandomForest model on the gene data
    """
    print("🧬 Loading gene expression data for disease classification...")
    try:
        X, y, preprocessor, original_df = load_data()
        
        # Create train/test split if Testing.csv doesn't exist
        sample_ids, X_test, y_test, test_preprocessor = load_test_data()
        if sample_ids is None:
            print("Creating train/test split from original data...")
            train_df, test_df = create_train_test_split(original_df)
            sample_ids = test_df["sample_id"]
            X_test, y_test, test_preprocessor = preprocess_data(test_df)
            
        # Count samples per class
        value_counts = pd.Series(y).value_counts()
        if len(value_counts) == 0:
            raise ValueError("No class labels found in the training data.")
            
        min_samples = min(value_counts)
        print(f"\nDisease type distribution:")
        for disease, count in value_counts.items():
            print(f"  - {disease}: {count} samples")
        
        n_splits = min(5, min_samples)
        
        # Define the classifier with tuned hyperparameters
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Create the feature selection step
        feature_selector = SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
        
        # Create the full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector),
            ('classifier', clf)
        ])
        
        # Perform cross-validation
        if n_splits >= 3:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')
            print(f"\n{n_splits}-fold CV Balanced Accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        else:
            print(f"\n⚠️ Not enough samples for proper CV (smallest class: {min_samples})")
            print("Proceeding with full training (no cross-validation)")
        
        # Fit the model on all training data
        print("\n🔬 Training model on all available data...")
        pipeline.fit(X, y)
        
        # Save the trained model
        joblib.dump(pipeline, "gene_classifier_model.pkl")
        print("✅ Model saved as 'gene_classifier_model.pkl'")
        
        # Test on unseen data
        print("\n🧪 Testing on unseen data...")
        
        # Use the pipeline to predict
        y_pred = pipeline.predict(X_test)
        
        # Display predictions
        print("\nPredictions:")
        for sample_id, pred in zip(sample_ids, y_pred):
            print(f"Sample {sample_id}: Predicted Disease Type → {pred}")
        
        # Show accuracy if labels exist
        if y_test is not None:
            test_acc = accuracy_score(y_test, y_pred)
            print(f"\n🎯 Accuracy on Testing data: {test_acc:.4f}")
            
            # Show detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Show confusion matrix
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
        else:
            print("\n⚠️ No true labels found in testing data. Skipping accuracy computation.")
        
        # Add predictions to original test data and save
        try:
            test_df = pd.read_csv("../data/Testing.csv")
            test_df["predicted_disease_type"] = y_pred
            test_df.to_csv("../data/Testing_with_predictions.csv", index=False)
            print("\n✅ Predictions saved to Testing_with_predictions.csv")
        except Exception as e:
            print(f"\n⚠️ Error saving predictions: {str(e)}")
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_feature_importance():
    """
    Analyze feature importance from the trained model
    """
    try:
        # Load the trained model
        pipeline = joblib.load("gene_classifier_model.pkl")
        
        # Get feature names after one-hot encoding
        preprocessor = pipeline.named_steps['preprocessor']
        feature_selector = pipeline.named_steps['feature_selection']
        clf = pipeline.named_steps['classifier']
        
        # Get feature names (this depends on the structure of your pipeline)
        try:
            # Try to get feature names from preprocessor
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback to generic feature names
            feature_names = [f'feature_{i}' for i in range(len(clf.feature_importances_))]
        
        # Get selected features
        selected_mask = feature_selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Get feature importances for selected features
        feature_importances = clf.feature_importances_
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\n🔍 Feature Importance Analysis:")
        print(importance_df.head(20))  # Show top 20 most important features
        
        # Save feature importance to CSV
        importance_df.to_csv("feature_importance.csv", index=False)
        print("✅ Feature importance saved to 'feature_importance.csv'")
        
    except Exception as e:
        print(f"\n⚠️ Error during feature importance analysis: {str(e)}")

if __name__ == "__main__":
    # Train the model
    train_model()
    
    # Analyze feature importance (only if model training succeeded)
    try:
        analyze_feature_importance()
    except Exception as e:
        print(f"Could not analyze feature importance: {str(e)}")