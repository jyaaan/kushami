import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for binary classification.
    gamma: Focuses on hard examples (higher = more focus).
    alpha: Balances the importance of positive and negative examples.
    """

    def loss_fn(y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Compute cross-entropy loss
        bce = -(
            alpha * y_true * tf.math.log(y_pred)
            + (1 - alpha) * (1 - y_true) * tf.math.log(1 - y_pred)
        )

        # Apply the gamma factor to emphasize hard examples
        loss = tf.math.pow(1 - y_pred, gamma) * bce
        return tf.reduce_mean(loss)

    return loss_fn


def build_model(input_shape=(128, 64, 3)):
    # V3? make model more complex
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),  # V1 adding more
            Flatten(),
            Dense(128, activation="relu"),  # magic? V1 increasing dense layer size
            Dropout(0.5),  # increase to address over-fitting
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        # loss=focal_loss(gamma=2.0, alpha=0.25),  # V2 Attempt to improve performance.
        metrics=["accuracy"],
    )
    return model
