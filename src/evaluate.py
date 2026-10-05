import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model and test data
model = load_model('models/model.h5')
x_test = np.load('data/processed/x_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Compute metrics
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

# Write metrics.json
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Generate confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")

print("Metrics saved to metrics.json and confusion matrix saved to models/")