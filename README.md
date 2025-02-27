# **MediAI: AI-Powered Risk Assessment & Health Advisory**

## **Introduction**
MediAI is an AI-powered medical diagnosis system designed to assess disease risk levels based on user health parameters. This project implements machine learning techniques to provide **personalized risk assessment** and **health recommendations** through an interactive web application.

## **Project Features**
✅ **Machine Learning-Based Risk Assessment** – Predicts Low, Moderate, or High risk levels.
✅ **Recommendation System** – Provides personalized health advice based on predictions.
✅ **User-Friendly Web Application** – Built using **Streamlit** for easy interaction.
✅ **Scalable & Deployable** – Ready for deployment on **Render, Heroku, or AWS**.

---

## **Project Structure**
```
AI_Medical_Diagnosis/
│── datasets/          # Store datasets here
│── models/            # Save trained models
│── notebooks/         # Jupyter notebooks for experiments
│── src/               # Python scripts (recommendation system, utilities)
│── web_app/           # Streamlit app files
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
```

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/jayadityadev/MediAI.git
cd MediAI
```

### **2. Set Up a Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate  # Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Jupyter Notebook (For Training & Preprocessing)**
```bash
jupyter notebook
```
Open `notebooks/Data_Preprocessing.ipynb` and `notebooks/Model_Training.ipynb` to process data and train the model.

### **5. Run the Web Application**
```bash
streamlit run web_app/app.py
```

---

## **How It Works**
1. **Data Preprocessing:** The dataset is cleaned, missing values handled, and features normalized.
2. **Machine Learning Model Training:** A **Random Forest Classifier** is trained to predict risk levels.
3. **Recommendation System:** Based on the predicted risk level, a personalized health recommendation is generated.
4. **Web Application:** Users input their health data, and the system predicts their risk level with recommendations.

---

## **Usage Guide**
1. Open the web application.
2. Enter your **Age, Glucose level, and BMI**.
3. Click **Check Risk Level**.
4. View your **Predicted Risk Level** and **Health Recommendation**.

---

## **Example Prediction**
Input:
```
Age: 50
Glucose: 140
BMI: 28
```
Output:
```
Predicted Risk Level: Moderate
Health Recommendation: Increase physical activity and monitor diet. Regular health checkups recommended.
```

---

## **Future Enhancements**
🔹 Expand dataset to include additional health indicators (blood pressure, cholesterol, etc.).
🔹 Improve recommendation system with AI-driven insights.
🔹 Deploy to a cloud platform for global accessibility.

---

## **License**
This project is open-source and available under the **MIT License**.

---

## **Contact**
For any questions or contributions, reach out via:
📧 Email: your.email@example.com  
🐙 GitHub: [yourusername](https://github.com/yourusername)