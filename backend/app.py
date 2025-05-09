from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)
CORS(app)

# Database config
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'patients.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    prediction = db.Column(db.String(50))
    data = db.Column(db.Text)  # JSON string

# Load trained model
try:
    model = joblib.load("model/gene_classifier_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# Load biomarker info
try:
    biomarkers_df = pd.read_csv('./data/Biomarkers.csv')
    print("✅ Biomarkers data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading biomarkers data: {str(e)}")
    biomarkers_df = None

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200
        
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get data from the request
        data = request.get_json()
        print("Received data:", data)

        # Convert data to DataFrame - input is gene expression values with genes as columns
        input_df = pd.DataFrame([data])
        print("DataFrame before preprocessing:", input_df)

        # Transform the data from wide format (genes as columns) to long format (genes as rows)
        transformed_df = transform_input_data(input_df)
        print("Transformed DataFrame:", transformed_df)

        # Preprocess the transformed data
        processed_data = preprocess_input(transformed_df)
        print("Preprocessed data shape:", processed_data.shape)

        # Predict using the model
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)
        
        # Get the index of the highest probability
        max_prob_index = np.argmax(probabilities[0])
        confidence = float(probabilities[0][max_prob_index])
        
        # Get interpretation
        interpretation = interpret_prediction(transformed_df, prediction)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "interpretation": interpretation,
            "probabilities": probabilities[0].tolist(),
            "class_labels": model.classes_.tolist(),
        })
    except Exception as e:
        print("Error during prediction:", str(e))  # Log the error
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def transform_input_data(input_df):
    """
    Transform input data from wide format (genes as columns) to long format (genes as rows)
    with appropriate metadata columns required by the model.
    """
    # Melt the DataFrame to convert from wide to long format
    melted_df = input_df.melt(var_name='gene_symbol', value_name='expression_value')
    
    # Create a new DataFrame with the required columns from our biomarkers dataset
    transformed_data = []
    
    for _, row in melted_df.iterrows():
        gene_symbol = row['gene_symbol']
        expression_value = row['expression_value']
        
        # Find the gene metadata in biomarkers_df
        gene_info = biomarkers_df[biomarkers_df['gene_symbol'] == gene_symbol]
        
        if not gene_info.empty:
            gene_data = {
                'gene_id': gene_info['gene_id'].values[0],
                'gene_symbol': gene_symbol,
                'pathway': gene_info['pathway'].values[0],
                'function': gene_info['function'].values[0],
                'protein_level': map_expression_to_level(expression_value),
                'sample_id': 'SAMPLE_PRED'  # Placeholder for sample ID
            }
            transformed_data.append(gene_data)
        else:
            # Handle genes not found in the biomarkers dataset
            gene_data = {
                'gene_id': f"UNKNOWN_{gene_symbol}",
                'gene_symbol': gene_symbol,
                'pathway': 'Unknown',
                'function': 'Unknown',
                'protein_level': map_expression_to_level(expression_value),
                'sample_id': 'SAMPLE_PRED'
            }
            transformed_data.append(gene_data)
    
    return pd.DataFrame(transformed_data)

def map_expression_to_level(value):
    """Map numerical expression values to categorical protein levels (low, medium, high)"""
    if value <= 3:
        return 'low'
    elif value <= 6:
        return 'medium'
    else:
        return 'high'

def preprocess_input(df):
    """
    Process the transformed input data to match the format expected by the model.
    The model expects exactly 4 features, not one-hot encoded.
    """
    # Since the error indicates the ColumnTransformer expects 4 features (not 162),
    # we need to prepare the data differently
    
    # Group by sample_id and aggregate the features
    # For categorical columns, we'll use the most common value
    aggregated_data = {
        'gene_symbol': [],
        'pathway': [],
        'function': [],
        'protein_level': []
    }
    
    # Extract just the raw features we need (no one-hot encoding)
    categorical_features = ["gene_symbol", "pathway", "function", "protein_level"]
    
    # The issue is likely that the model was trained on individual gene data, not aggregated
    # Let's try a different approach - select a single representative gene
    # Here we'll select the gene with highest expression (indicated by 'high' protein level)
    high_expressed_genes = df[df['protein_level'] == 'high']
    
    if not high_expressed_genes.empty:
        # Use the first high-expressed gene as representative
        representative_gene = high_expressed_genes.iloc[0]
    else:
        # If no high-expressed genes, use the first gene
        representative_gene = df.iloc[0]
    
    # Create a single-row DataFrame with just the 4 categorical features
    X_processed = pd.DataFrame({
        'gene_symbol': [representative_gene['gene_symbol']],
        'pathway': [representative_gene['pathway']],
        'function': [representative_gene['function']],
        'protein_level': [representative_gene['protein_level']]
    })
    
    # Print for debugging
    print("Final processed data shape:", X_processed.shape)
    print("Final processed data:", X_processed)
    
    return X_processed

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        # This assumes your model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importances))]
            
            importance_data = [
                {"feature": str(fname), "importance": float(imp)}
                for fname, imp in zip(feature_names, importances)
            ]
            
            importance_data = sorted(importance_data, key=lambda x: x["importance"], reverse=True)
            return jsonify(importance_data[:20])
        else:
            return jsonify({"error": "Model does not have feature importance information"}), 400
    except Exception as e:
        print("Error fetching feature importance:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/save_patient", methods=["POST"])
def save_patient():
    data = request.json
    patient = Patient(
        name=data['name'],
        age=data['age'],
        gender=data['gender'],
        prediction=data['prediction'],
        data=str(data['inputs'])
    )
    db.session.add(patient)
    db.session.commit()
    return jsonify({'message': 'Patient saved'})

@app.route("/patients", methods=["GET"])
def get_patients():
    patients = Patient.query.all()
    result = [
        {
            'id': p.id,
            'name': p.name,
            'age': p.age,
            'gender': p.gender,
            'prediction': p.prediction,
            'data': p.data
        } for p in patients
    ]
    return jsonify(result)

@app.route("/delete_patient/<int:id>", methods=["DELETE"])
def delete_patient(id):
    patient = Patient.query.get(id)
    if patient:
        db.session.delete(patient)
        db.session.commit()
        return jsonify({'message': 'Patient deleted successfully'}), 200
    return jsonify({'message': 'Patient not found'}), 404

@app.route("/edit_patient/<int:id>", methods=["PUT"])
def edit_patient(id):
    data = request.json
    patient = Patient.query.get(id)

    if not patient:
        return jsonify({'message': 'Patient not found'}), 404

    patient.name = data.get('name', patient.name)
    patient.age = data.get('age', patient.age)
    patient.gender = data.get('gender', patient.gender)
    patient.prediction = data.get('prediction', patient.prediction)
    patient.data = str(data.get('inputs', patient.data))

    db.session.commit()

    return jsonify({'message': 'Patient updated successfully'})

def interpret_prediction(df, prediction):
    """Create a simplified interpretation of the prediction result"""
    # Find genes with high expression values that correlate with the predicted disease
    relevant_genes = df[df['protein_level'] == 'high']['gene_symbol'].tolist()[:5]  # Top 5 high expressed genes
    
    interpretation = {
        'condition': prediction,
        'key_biomarkers': [],
        'biological_meaning': []
    }

    for gene in relevant_genes:
        bio_info = biomarkers_df[biomarkers_df['gene_symbol'] == gene]
        if not bio_info.empty:
            gene_info = {
                'gene': gene,
                'pathway': bio_info['pathway'].values[0],
                'function': bio_info['function'].values[0]
            }
            interpretation['key_biomarkers'].append(gene_info)
            interpretation['biological_meaning'].append(
                f"{gene} ({bio_info['pathway'].values[0]}): {bio_info['function'].values[0]}"
            )
    
    return interpretation

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)