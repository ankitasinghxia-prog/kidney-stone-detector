import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random

print("🚀 Starting Kidney Stone Detection...")

class UltraSimpleKidneyDetector:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None
        self.training_history = None
    
    def create_generators(self, data_dir):
        """Create data generators for memory-efficient loading"""
        
        print(f"📁 Looking for dataset in: {data_dir}")
        
        # Count images first
        stone_folder = os.path.join(data_dir, 'stone')
        normal_folder = os.path.join(data_dir, 'normal')
        
        stone_count = len([f for f in os.listdir(stone_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(stone_folder) else 0
        normal_count = len([f for f in os.listdir(normal_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(normal_folder) else 0
        
        print(f"🔍 Found {stone_count} stone images")
        print(f"🔍 Found {normal_count} normal images")
        
        if stone_count == 0 and normal_count == 0:
            print("❌ No images found! Creating demo data...")
            return None, None, None, None
        
        # Create data generators (loads images on-the-fly, not all at once!)
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator, stone_count, normal_count
    
    def create_model(self):
        """Create CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def train_simple(self, train_generator, val_generator, epochs=20):
        """Train the model using generators"""
        
        if train_generator is None:
            print("❌ No data to train on!")
            return None
        
        # Create model
        self.model = self.create_model()
        
        print("\n🧠 Training...")
        
        # Train
        self.training_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_steps=val_generator.samples // val_generator.batch_size
        )
        
        # Save model
        self.model.save('kidney_stone_model.h5')
        print("💾 Model saved as 'kidney_stone_model.h5'")
        
        # Plot training history
        self.plot_history()
        
        return self.training_history
    
    def plot_history(self):
        """Plot training accuracy and loss"""
        if self.training_history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.training_history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.training_history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.training_history.history['loss'], label='Train Loss')
        ax2.plot(self.training_history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("📊 Training history saved as 'training_history.png'")


if __name__ == "__main__":
    print("="*60)
    print("🏥 KIDNEY STONE DETECTION SYSTEM")
    print("="*60)
    
    detector = UltraSimpleKidneyDetector()
    
    # Create generators (memory efficient!)
    train_gen, val_gen, stone_count, normal_count = detector.create_generators('dataset')
    
    if train_gen is not None:
        detector.train_simple(train_gen, val_gen, epochs=20)
        print("\n🎉 Training completed!")
        print("🚀 Run: streamlit run app.py")
    else:
        print("❌ Please add images to dataset/stone and dataset/normal folders")