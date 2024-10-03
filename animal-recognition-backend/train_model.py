import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Load the MobileNetV2 model with pre-trained ImageNet weights, exclude the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (this keeps the pre-trained layers intact)
base_model.trainable = False

# Add custom layers for dog breed classification (Stanford Dogs has 120 breeds)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(120, activation='softmax')  # 120 classes for 120 breeds
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to see the architecture
model.summary()

# Set up ImageDataGenerator for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                   horizontal_flip=True, fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories (replace 'stanford_dogs/train' and 'stanford_dogs/validation' with your dataset paths)
train_generator = train_datagen.flow_from_directory(
    'stanford_dogs/train',  # Path to your training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'stanford_dogs/validation',  # Path to your validation data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model on the dataset
history = model.fit(
    train_generator,
    epochs=10,  # Adjust based on your needs
    validation_data=validation_generator
)

# Save the fine-tuned model for use in your Flask API
model.save("animal_recognition_model.h5")
