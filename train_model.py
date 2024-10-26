import matplotlib.pyplot as plt


from data_generator import create_generators
from model import build_model

"""
pre-tuning:
Classification Report:
              precision    recall  f1-score   support

  Background       0.92      0.92      0.92       681
      Sneeze       0.04      0.04      0.04        56

    accuracy                           0.85       737
   macro avg       0.48      0.48      0.48       737
weighted avg       0.85      0.85      0.85       737

Confusion Matrix:
[[626  55]
 [ 54   2]]

 wow! bad!

 post-tuning

 """


def train():
    train_dataset, val_dataset = create_generators()
    model = build_model()

    # V2 Attempt to improve performance
    # Heavily weight sneezes (class 1)
    class_weights = {0: 1.0, 1: 10.0}

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        class_weight=class_weights,
    )

    model.save("sneeze_detector.keras")
    print("Model saved as sneeze_detector.keras")
    return history


if __name__ == "__main__":
    history = train()
    # Plot accuracy
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # benchmarks
    """ 
    for binary classification
    mnist - look into this, toy problem set for classifying hand written digits
       """
