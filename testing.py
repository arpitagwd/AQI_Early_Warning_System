import joblib
le_city = joblib.load("models/city_encoder.pkl")
print(le_city.classes_)