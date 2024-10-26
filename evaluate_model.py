from tensorflow.keras.models import load_model
from data_generator import create_generators
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate():
    model = load_model("sneeze_detector.keras")
    _, val_dataset = create_generators()

    # Predictions
    y_true = np.concatenate([y for _, y in val_dataset], axis=0)  # True labels
    y_pred_probs = model.predict(val_dataset)  # predicted probabilities
    y_pred = (y_pred_probs > 0.5).astype(int)  # convert probabilities to predictions

    # metrics!
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Background", "Sneeze"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
