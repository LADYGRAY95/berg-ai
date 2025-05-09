import React, { useState, useEffect } from 'react';
import { FaEdit, FaTrashAlt } from 'react-icons/fa';

const featureNames = [
  "MDM2", "CDKN1A", "BAX", "WNT1", "FZD1",
  "BRCA1", "BRCA2", "CDKN2A", "EGFR", "ERBB2",
  "TP53", "RB1", "ATM", "MLH1", "MSH2",
  "APC", "KRAS", "NRAS", "PIK3CA", "PTEN",
  "SMAD4", "VHL", "ABL1", "AKT1", "ALK"
];

const Dashboard = () => {
  const [patients, setPatients] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  const [currentPatient, setCurrentPatient] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      const response = await fetch('http://127.0.0.1:5000/patients');
      const data = await response.json();
      setPatients(data);
    };
    fetchPatients();
  }, []);

  const handleEdit = (id) => {
    const patientToEdit = patients.find(patient => patient.id === id);
    if (patientToEdit) {
      setCurrentPatient(patientToEdit); // Set the patient data in the form
      setIsEditing(true); // Open the edit form
    } else {
      console.error('Patient not found');
    }
  };

  const handleDelete = async (id) => {
    const res = await fetch(`http://127.0.0.1:5000/delete_patient/${id}`, { method: 'DELETE' });
    const data = await res.json();
    alert(data.message); // Show a confirmation message
    setPatients(patients.filter(patient => patient.id !== id)); // Remove the patient from the list
  };

  const handleSaveEdit = async () => {
    const updatedPatient = {
      id: currentPatient.id,
      name: currentPatient.name,
      age: currentPatient.age,
      gender: currentPatient.gender,
      genes: currentPatient.genes, // Include the gene data in the update
    };

    const res = await fetch('http://127.0.0.1:5000/update_patient', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updatedPatient),
    });

    const data = await res.json();
    alert(data.message);

    setPatients(patients.map(patient =>
      patient.id === currentPatient.id ? updatedPatient : patient
    ));

    setIsEditing(false); // Close the edit form
    setCurrentPatient(null); // Reset the form
  };

  const handleGeneChange = (e, gene) => {
    setCurrentPatient({
      ...currentPatient,
      genes: {
        ...currentPatient.genes,
        [gene]: e.target.value
      }
    });
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>📊 Patient Dashboard</h1>

      {isEditing && currentPatient ? (
        <div style={styles.editForm}>
          <h2>Edit Patient</h2>
          <form onSubmit={e => e.preventDefault()}>
            {/* General Patient Info */}
            <input
              type="text"
              value={currentPatient.name || ""}
              onChange={e => setCurrentPatient({ ...currentPatient, name: e.target.value })}
              placeholder="Name"
            />
            <input
              type="number"
              value={currentPatient.age || ""}
              onChange={e => setCurrentPatient({ ...currentPatient, age: e.target.value })}
              placeholder="Age"
            />
            <select
              value={currentPatient.gender || ""}
              onChange={e => setCurrentPatient({ ...currentPatient, gender: e.target.value })}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>

            {/* Gene Data */}
            <h3>Gene Expression Data</h3>
            {featureNames.map(gene => (
              <div key={gene}>
                <label>{gene}</label>
                <input
                  type="number"
                  value={currentPatient.genes[gene] || ""}
                  onChange={(e) => handleGeneChange(e, gene)}
                  placeholder={gene}
                  step="any"
                />
              </div>
            ))}

            <button type="button" onClick={handleSaveEdit}>Save</button>
            <button type="button" onClick={() => setIsEditing(false)}>Cancel</button>
          </form>
        </div>
      ) : (
        <table style={styles.table}>
          <thead>
            <tr>
              <th>Name</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Prediction</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {patients.length === 0 ? (
              <tr>
                <td colSpan="5" style={styles.noPatients}>No patients found.</td>
              </tr>
            ) : (
              patients.map((patient) => (
                <tr key={patient.id}>
                  <td>{patient.name}</td>
                  <td>{patient.age}</td>
                  <td>{patient.gender}</td>
                  <td>{patient.prediction}</td>
                  <td style={styles.actions}>
                    <FaEdit
                      onClick={() => handleEdit(patient.id)}
                      style={styles.icon}
                      title="Edit"
                    />
                    <FaTrashAlt
                      onClick={() => handleDelete(patient.id)}
                      style={styles.icon}
                      title="Delete"
                    />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      )}
    </div>
  );
};

const styles = {
  container: {
    fontFamily: 'Segoe UI, sans-serif',
    maxWidth: '960px',
    margin: '0 auto',
    padding: '2rem',
    backgroundColor: '#f5f7fb',
    color: '#333',
    borderRadius: '8px',
    boxShadow: '0 8px 20px rgba(0,0,0,0.1)',
  },
  heading: {
    textAlign: 'center',
    color: '#3a3a7c',
    marginBottom: '1.5rem',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: '1rem',
  },
  th: {
    padding: '0.5rem',
    textAlign: 'left',
    borderBottom: '2px solid #ddd',
  },
  td: {
    padding: '0.5rem',
    borderBottom: '1px solid #ddd',
  },
  actions: {
    display: 'flex',
    gap: '1rem',
    justifyContent: 'center',
  },
  icon: {
    cursor: 'pointer',
    color: '#3a3a7c',
    fontSize: '1.25rem',
    transition: 'color 0.3s',
  },
  iconHover: {
    color: '#ff6347', // Red for delete
  },
  editForm: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    marginBottom: '1rem',
  },
  noPatients: {
    textAlign: 'center',
    fontSize: '1.2rem',
    color: '#888',
  },
};

export default Dashboard;