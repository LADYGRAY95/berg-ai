# BERG AI: Centric Disease Classification Using Gene Expression

## 📌 Overview
BERG AI is an AI-powered classification model designed to predict disease types from gene expression data. It provides a user-friendly web application for medical researchers, students, and health tech developers to classify diseases accurately and efficiently. The system is trained on data for 7 disease categories and offers real-time predictions.

## 🎯 Problem Statement
- Medical diagnosis often lacks early and accurate detection.
- Traditional methods are time-consuming and rely on visible symptoms.
- Misdiagnosis or delayed diagnosis can lead to severe consequences.

## ✨ Solution
BERG AI leverages machine learning to:
- Predict disease types from gene expression data.
- Provide a simple, intuitive interface for users to input data and view results.
- Support research, diagnostics, and precision medicine efforts.

## 🔧 Key Features
- **Manual Input**: Users can input patient gene expression values via a form.
- **Real-Time Classification**: Predicts one of 7 disease categories instantly.
- **Patient Dashboard**: View, edit, or delete patient records.
- **Batch Processing**: Future support for CSV uploads (planned feature).

## 🛠️ Technologies Used
- **Frontend**: React + Vite  
- **Backend**: Flask REST API  
- **Model**: Random Forest Classifier (scikit-learn)  
- **Data Format**: .csv files  
- **Pipeline**: Upload → Predict → Display  

## 📊 Model Performance
- **Training Samples**: 3000  
- **Cross-Validation Accuracy**: 100% (5-fold)  
- **Testing Accuracy**: 76.00%  

## 👥 Target Audience
- Medical researchers managing datasets.  
- Students in bioinformatics or health tech projects.  
- Health tech developers testing ML-based disease classifiers.  

## 🚀 Use Cases
- Research diagnostics.  
- Genetic counseling support.  
- Drug development targeting.  

## 🔮 Future Enhancements
- **CSV Upload**: Batch input for multiple patients.  
- **OCR Integration**: Extract gene expressions from scanned documents.  
- **User Authentication**: Role-based access for medical specialists.  
- **Cloud Integration**: Secure data storage and multi-device accessibility.  
- **Mobile Version**: Expand access to smartphones.  

## 🌟 Impact & Ethics
- Enables faster, non-invasive preliminary diagnosis.  
- Improves research capabilities.  
- **Challenges**:  
  - Real-world dataset variability.  
  - Medical data privacy concerns.  
  - Clinical validation requirements.  

## 📂 Corrected Project Structure
```
berg_ai/
├── backend/           # Flask REST API
│   ├── model/        # Contains model and data
│   │   ├── train_model.py
│   │── [data files]
│   └── app.py        # Main backend application
└── frontend/         # Main frontend folder
    └── frontend/     # React application (actual frontend code)
        ├── public/
        └── src/
```

## 📝 How to Run

### 1. Train the Model
```bash
cd backend/model
python train_model.py
```

### 2. Start the Backend Server
```bash
cd backend
python app.py
```

### 3. Set Up and Run the Frontend
```bash
cd frontend/frontend
npm install
npm run dev
```

## 📞 Contact
- **Youssr Chouaya**  
  - Email: [cyoussr@gmail.com](mailto:cyoussr@gmail.com)  
  - GitHub: [LADYGRAY95](https://github.com/LADYGRAY95)  
  - LinkedIn: [Youssr Chouaya](https://linkedin.com/in/youssr-chouaya-63a54929a)  

---
**Note**: The application will be available at `http://localhost:3000` after starting the frontend. Make sure both backend and frontend servers are running for full functionality. Contributions and feedback are welcome! 🚀
