import joblib
import numpy as np

# Wczytanie modelu
model = joblib.load("model_v1.joblib")

# Przykładowe dane wejściowe (możesz zmienić wartości)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# Wykonanie predykcji
prediction = model.predict(sample_input)
predicted_class = np.argmax(prediction)

print(f"Predykcja modelu: {prediction}")
print(f"To jest klasa: {predicted_class}")