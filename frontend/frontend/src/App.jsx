import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const featureNames = [
  "MDM2", "CDKN1A", "BAX", "WNT1", "FZD1",
  "BRCA1", "BRCA2", "CDKN2A", "EGFR", "ERBB2",
  "TP53", "RB1", "ATM", "MLH1", "MSH2",
  "APC", "KRAS", "NRAS", "PIK3CA", "PTEN",
  "SMAD4", "VHL", "ABL1", "AKT1", "ALK"
];

function App() {
  const [formData, setFormData] = useState(
    featureNames.reduce((acc, name) => ({ ...acc, [name]: '' }), {})
  );
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [patientInfo, setPatientInfo] = useState({ name: '', age: '', gender: '' });

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);

    const emptyFields = featureNames.filter(name => formData[name] === '');
    if (emptyFields.length > 0) {
      alert(`Please fill in all the needed information.`);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(formData).map(([k, v]) => [k, parseFloat(v) || 0])
          )
        ),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(featureNames.reduce((acc, name) => ({ ...acc, [name]: '' }), {}));
    setResult(null);
    setPatientInfo({ name: '', age: '', gender: '' });
  };

  const handleSave = async () => {
    if (!result) return;
    const saveData = {
      name: patientInfo.name,
      age: patientInfo.age,
      gender: patientInfo.gender,
      prediction: result.prediction,
      inputs: formData
    };

    const res = await fetch('http://127.0.0.1:5000/save_patient', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(saveData)
    });
    const data = await res.json();
    alert(data.message);
  };

  const handleViewDashboard = () => {
    navigate('/dashboard');
  };

  useEffect(() => {
    if (result) console.log("✅ Prediction Result:", result);
  }, [result]);

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>🧬 Gene Expression Predictor</h1>

      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ color: '#3a3a7c' }}>🧑 Patient Info</h2>
        <input
          type="text"
          placeholder="Name"
          value={patientInfo.name}
          onChange={e => setPatientInfo({ ...patientInfo, name: e.target.value })}
          style={styles.input}
        />
        <input
          type="number"
          placeholder="Age"
          value={patientInfo.age}
          onChange={e => setPatientInfo({ ...patientInfo, age: e.target.value })}
          style={styles.input}
        />
        <select
          value={patientInfo.gender}
          onChange={e => setPatientInfo({ ...patientInfo, gender: e.target.value })}
          style={styles.input}
        >
          <option value="">Select Gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
          <option value="Other">Other</option>
        </select>
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.grid}>
          {featureNames.map((name) => (
            <div key={name} style={styles.inputGroup}>
              <label style={styles.label}>{name}</label>
              <input
                type="number"
                name={name}
                value={formData[name]}
                onChange={handleChange}
                step="any"
                style={styles.input}
              />
            </div>
          ))}
        </div>
        <button type="submit" style={styles.button}>
          {loading ? 'Predicting...' : '🔍 Predict'}
        </button>
      </form>

      <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
        <button type="button" style={styles.button} onClick={handleReset}>🔄 Reset</button>
        <button type="button" style={styles.button} onClick={handleSave}>💾 Save</button>
        <button type="button" style={styles.button} onClick={handleViewDashboard}>📊 Dashboard</button>
      </div>

      {result && (
        <div style={styles.result}>
          {result.error ? (
            <p style={styles.error}>⚠️ Error: {result.error}</p>
          ) : (
            <>
              <h2 style={styles.prediction}>
                Result: {result.prediction?.toUpperCase() || 'N/A'}
              </h2>

              {result.confidence && (
                <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
              )}

              {result.class_labels && result.probabilities && (
                <div>
                  <h3 style={styles.subheading}>📊 Probabilities</h3>
                  <ul style={styles.importanceList}>
                    {result.class_labels.map((label, i) => (
                      <li key={label}>
                        <strong>{label}:</strong> {(result.probabilities[i] * 100).toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.interpretation?.key_biomarkers?.length > 0 && (
                <>
                  <h3 style={styles.subheading}>🧠 Interpretation</h3>
                  <ul style={styles.importanceList}>
                    {result.interpretation.key_biomarkers.map((bio, idx) => (
                      <li key={idx} style={styles.importanceItem}>
                        <strong>{bio.gene}</strong><br />
                        Pathway: {bio.pathway}<br />
                        Function: {bio.function}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    fontFamily: 'Segoe UI, sans-serif',
    maxWidth: '960px',
    margin: '0 auto',
    padding: '2rem',
    backgroundColor: '#f5f7fb',
    color: '#333',
    borderRadius: '8px',
    boxShadow: '0 8px 20px rgba(0,0,0,0.1)'
  },
  heading: {
    textAlign: 'center',
    color: '#3a3a7c',
    marginBottom: '1.5rem'
  },
  form: {
    display: 'flex',
    flexDirection: 'column'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: '1rem',
    marginBottom: '2rem'
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column'
  },
  label: {
    fontSize: '0.85rem',
    marginBottom: '0.25rem',
    color: '#555'
  },
  input: {
    padding: '0.4rem 0.6rem',
    borderRadius: '4px',
    border: '1px solid #ccc',
    backgroundColor: '#fff',
    fontSize: '1rem'
  },
  button: {
    padding: '0.75rem',
    fontSize: '1rem',
    backgroundColor: '#3a3a7c',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease'
  },
  result: {
    marginTop: '2rem',
    padding: '1rem',
    borderRadius: '6px',
    backgroundColor: '#fff',
    boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
  },
  prediction: {
    fontSize: '1.5rem',
    color: '#3a3a7c'
  },
  subheading: {
    marginTop: '1rem',
    color: '#444'
  },
  importanceList: {
    listStyle: 'none',
    paddingLeft: '0',
    marginTop: '0.5rem'
  },
  importanceItem: {
    marginBottom: '0.5rem',
    fontSize: '0.95rem'
  },
  error: {
    color: '#d33',
    fontWeight: 'bold'
  }
};

export default App;