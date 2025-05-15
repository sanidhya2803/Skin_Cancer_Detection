import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import cv2

# Model definition
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: Benign, Malignant
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        '/content/dataset/dataset/train',  # Path to train directory
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        '/content/dataset/dataset/test',  # Path to test directory
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    
    model = build_model()
    model.fit(train_generator, epochs=10, validation_data=val_generator)
    model.save('melanoma_classification_model.h5')

# Prediction function for the web application
def predict_skin_cancer(image_path):
    model = tf.keras.models.load_model('melanoma_classification_model.h5')
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    
    labels = ['Benign', 'Malignant']
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

# Evaluation metrics for the test set
def evaluate_model():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        '/content/dataset/dataset/test',  # Path to test directory
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    
    model = tf.keras.models.load_model('melanoma_classification_model.h5')
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)
    
    # Convert one-hot encoded labels to class labels
    true_labels = np.argmax(test_labels, axis=1)
    pred_labels = np.argmax(predictions, axis=1)

    # Calculating evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_report = classification_report(true_labels, pred_labels)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

# Call train_model to train, and evaluate_model to test on the test dataset
if __name__ == "__main__":
    train_model()  # Trains the model
    evaluate_model()  # Evaluates on the test dataset
