"""
Advanced Rice Classification Model Training Script
This script creates and trains a more sophisticated model for rice classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RiceClassificationModel:
    def __init__(self, num_classes=5, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create the rice classification model using transfer learning"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Add custom classification layers
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_categorical_accuracy']
        )
        
        return self.model
    
    def create_data_generators(self, train_dir=None, validation_dir=None):
        """Create data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Use 20% for validation if no separate validation dir
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        if train_dir and os.path.exists(train_dir):
            # If you have actual data directories
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )
        else:
            # Create synthetic data for demonstration
            print("Creating synthetic training data for demonstration...")
            train_generator, validation_generator = self.create_synthetic_data()
        
        return train_generator, validation_generator
    
    def create_synthetic_data(self):
        """Create synthetic data for demonstration purposes"""
        # Generate synthetic rice grain images
        def generate_rice_images(num_samples=1000):
            images = []
            labels = []
            
            for i in range(num_samples):
                # Create synthetic rice grain-like images
                img = np.random.rand(224, 224, 3)
                
                # Add some rice grain-like features
                center_x, center_y = np.random.randint(50, 174, 2)
                
                # Create oval-like shapes (rice grains are typically oval)
                y, x = np.ogrid[:224, :224]
                mask = ((x - center_x) / 30) ** 2 + ((y - center_y) / 60) ** 2 <= 1
                
                # Apply rice-like coloring
                rice_color = np.random.choice([
                    [0.9, 0.9, 0.8],  # White rice
                    [0.8, 0.7, 0.6],  # Brown rice
                    [0.9, 0.8, 0.7],  # Basmati-like
                    [0.85, 0.8, 0.75], # Jasmine-like
                    [0.88, 0.85, 0.8]  # Arborio-like
                ])
                
                img[mask] = rice_color + np.random.normal(0, 0.1, 3)
                img = np.clip(img, 0, 1)
                
                images.append(img)
                labels.append(i % self.num_classes)
            
            return np.array(images), tf.keras.utils.to_categorical(labels, self.num_classes)
        
        # Generate training and validation data
        train_images, train_labels = generate_rice_images(800)
        val_images, val_labels = generate_rice_images(200)
        
        # Create data generators
        train_generator = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_generator = train_generator.batch(32).prefetch(tf.data.AUTOTUNE)
        
        validation_generator = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        validation_generator = validation_generator.batch(32).prefetch(tf.data.AUTOTUNE)
        
        return train_generator, validation_generator
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'best_rice_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("Starting model training...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning: Unfreeze some layers of the base model
        print("Fine-tuning the model...")
        self.model.layers[0].trainable = True
        
        # Freeze the first 100 layers
        for layer in self.model.layers[0].layers[:100]:
            layer.trainable = False
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_categorical_accuracy']
        )
        
        # Continue training
        fine_tune_epochs = 20
        self.history_fine = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, filepath='rice_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Training Loss')
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.show()

def main():
    """Main training function"""
    print("🌾 Rice Classification Model Training")
    print("=" * 50)
    
    # Create model instance
    rice_model = RiceClassificationModel()
    
    # Create the model
    model = rice_model.create_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create data generators
    train_gen, val_gen = rice_model.create_data_generators()
    
    # Train the model
    rice_model.train_model(train_gen, val_gen, epochs=30)
    
    # Save the model
    rice_model.save_model('rice_model.h5')
    
    # Plot training history
    rice_model.plot_training_history()
    
    print("✅ Training completed successfully!")
    print("Model saved as 'rice_model.h5'")

if __name__ == "__main__":
    main()
