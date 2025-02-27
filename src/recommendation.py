def get_recommendation(risk_level):
    if risk_level == "Low":
        return "Maintain a healthy lifestyle with regular exercise and a balanced diet."
    elif risk_level == "Moderate":
        return "Increase physical activity and monitor diet. Regular health checkups recommended."
    else:
        return "High risk! Consult a doctor immediately and follow medical advice."

