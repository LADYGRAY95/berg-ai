import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

def visualize_dataset(file_path='../data/Training.csv'):
    """
    Create visualizations for the gene dataset
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Set up the figure
        plt.figure(figsize=(16, 12))
        plt.suptitle(f'Gene Dataset Analysis: {file_path}', fontsize=16)
        
        # 1. Disease type distribution
        plt.subplot(2, 2, 1)
        disease_counts = df['disease_type'].value_counts()
        sns.barplot(x=disease_counts.index, y=disease_counts.values)
        plt.title('Disease Type Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # 2. Pathway distribution
        plt.subplot(2, 2, 2)
        pathway_counts = df['pathway'].value_counts().head(10)  # Top 10 pathways
        sns.barplot(x=pathway_counts.index, y=pathway_counts.values)
        plt.title('Top 10 Pathway Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # 3. Protein level distribution
        plt.subplot(2, 2, 3)
        protein_level_counts = df['protein_level'].value_counts()
        sns.barplot(x=protein_level_counts.index, y=protein_level_counts.values)
        plt.title('Protein Level Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # 4. Pathway by disease type (heatmap)
        plt.subplot(2, 2, 4)
        pathway_disease = pd.crosstab(df['pathway'], df['disease_type'])
        sns.heatmap(pathway_disease, cmap='viridis', annot=True, fmt='d', cbar=True)
        plt.title('Pathway by Disease Type')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('gene_dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Dataset visualizations saved as 'gene_dataset_analysis.png'")
        
        # Additional visualization: Disease type by protein level
        plt.figure(figsize=(10, 6))
        disease_protein = pd.crosstab(df['disease_type'], df['protein_level'])
        sns.heatmap(disease_protein, cmap='YlGnBu', annot=True, fmt='d', cbar=True)
        plt.title('Disease Type by Protein Level')
        plt.tight_layout()
        plt.savefig('disease_by_protein_level.png', dpi=300, bbox_inches='tight')
        print("✅ Disease by protein level visualization saved as 'disease_by_protein_level.png'")
        
    except Exception as e:
        print(f"Error visualizing dataset: {str(e)}")

def visualize_feature_importance(feature_file='feature_importance.csv', top_n=20):
    """
    Visualize feature importance from the model
    """
    try:
        # Load feature importance data
        importance_df = pd.read_csv(feature_file)
        
        # Get top N features
        top_features = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig('top_features_importance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Top {top_n} features visualization saved as 'top_features_importance.png'")
        
    except Exception as e:
        print(f"Error visualizing feature importance: {str(e)}")

def visualize_dimensionality_reduction(file_path='../data/Training.csv'):
    """
    Visualize data using dimensionality reduction techniques (PCA and t-SNE)
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # One-hot encode categorical features
        categorical_features = ['pathway', 'function', 'protein_level']
        encoded_df = pd.get_dummies(df[categorical_features])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(encoded_df)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)//2))
        tsne_result = tsne.fit_transform(encoded_df)
        
        # Create figure
        plt.figure(figsize=(16, 7))
        
        # Plot PCA
        plt.subplot(1, 2, 1)
        for disease_type in df['disease_type'].unique():
            indices = df['disease_type'] == disease_type
            plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=disease_type, alpha=0.7)
        plt.title('PCA: Gene Features by Disease Type')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        
        # Plot t-SNE
        plt.subplot(1, 2, 2)
        for disease_type in df['disease_type'].unique():
            indices = df['disease_type'] == disease_type
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=disease_type, alpha=0.7)
        plt.title('t-SNE: Gene Features by Disease Type')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        print("✅ Dimensionality reduction visualization saved as 'dimensionality_reduction.png'")
        
    except Exception as e:
        print(f"Error in dimensionality reduction: {str(e)}")

def visualize_prediction_results(prediction_file='../data/Testing_with_predictions.csv'):
    """
    Visualize prediction results
    """
    try:
        # Load predictions
        df = pd.read_csv(prediction_file)
        
        if 'disease_type' not in df.columns or 'predicted_disease_type' not in df.columns:
            print("⚠️ Required columns 'disease_type' and/or 'predicted_disease_type' not found.")
            return
        
        # Create confusion matrix
        plt.figure(figsize=(10, 8))
        cm = pd.crosstab(df['disease_type'], df['predicted_disease_type'], 
                        rownames=['True'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
        
        # Calculate accuracy by disease type
        accuracy_by_type = {}
        for disease in df['disease_type'].unique():
            disease_df = df[df['disease_type'] == disease]
            correct = sum(disease_df['disease_type'] == disease_df['predicted_disease_type'])
            accuracy_by_type[disease] = correct / len(disease_df)
        
        # Plot accuracy by disease type
        plt.figure(figsize=(10, 6))
        diseases = list(accuracy_by_type.keys())
        accuracies = list(accuracy_by_type.values())
        sns.barplot(x=diseases, y=accuracies)
        plt.title('Prediction Accuracy by Disease Type')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('accuracy_by_disease.png', dpi=300, bbox_inches='tight')
        print("✅ Accuracy by disease type saved as 'accuracy_by_disease.png'")
        
    except Exception as e:
        print(f"Error visualizing prediction results: {str(e)}")

if __name__ == "__main__":
    # Visualize the dataset
    visualize_dataset('../data/Training.csv')
    
    # Try to visualize feature importance if file exists
    try:
        visualize_feature_importance('feature_importance.csv')
    except:
        print("⚠️ Feature importance file not found. Run the model first.")
    
    # Dimensionality reduction visualization
    visualize_dimensionality_reduction('../data/Training.csv')
    
    # Try to visualize prediction results if file exists
    try:
        visualize_prediction_results('../data/Testing_with_predictions.csv')
    except:
        print("⚠️ Prediction results file not found. Run the model first.")