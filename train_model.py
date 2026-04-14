import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("🚀 Starting Ultra-Simplified Kidney Stone Detection...")

class UltraSimpleKidneyDetector:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None
    
    def load_images_simple(self, data_dir):
        """Super simple image loading without complex dependencies"""
        images = []
        labels = []
        
        print(f"📁 Looking for dataset in: {data_dir}")
        
        stone_folder = os.path.join(data_dir, 'stone')
        if os.path.exists(stone_folder):
            stone_files = [f for f in os.listdir(stone_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"🔍 Found {len(stone_files)} stone images")
            
            for img_file in stone_files[:50]:  
                try:
                    img_path = os.path.join(stone_folder, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(1)  
                except Exception as e:
                    print(f"⚠️ Error loading {img_file}: {e}")
        
        
        normal_folder = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_folder):
            normal_files = [f for f in os.listdir(normal_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"🔍 Found {len(normal_files)} normal images")
            
            for img_file in normal_files[:50]:  
                try:
                    img_path = os.path.join(normal_folder, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(0)  
                except Exception as e:
                    print(f"⚠️ Error loading {img_file}: {e}")
        
        if len(images) == 0:
            print("❌ No images found! Creating demo data...")

            for i in range(20):
                dummy_img = np.random.random((*self.img_size, 3))
                images.append(dummy_img)
                labels.append(i % 2)
        
        print(f"✅ Total images loaded: {len(images)}")
        return np.array(images), np.array(labels)
    
    def create_ultra_simple_model(self):
        """Create a very simple model"""
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_simple(self, X, y, epochs=5):
        """Simple training without complex splits"""
        if len(X) == 0:
            print("❌ No data to train on!")
            return
        

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Simple split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        self.model = self.create_ultra_simple_model()
        
        print("🧠 Training simple model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=8,
            verbose=1
        )
        
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test accuracy: {test_acc:.2%}")
        
    
        self.model.save('kidney_stone_model.h5')
        print("💾 Model saved as 'kidney_stone_model.h5'")
        
        return history


if __name__ == "__main__":
    print("=" * 50)
    print("🏥 KIDNEY STONE DETECTION - SIMPLIFIED VERSION")
    print("=" * 50)
    
    detector = UltraSimpleKidneyDetector()
    
    
    X, y = detector.load_images_simple('dataset')
    
    
    detector.train_simple(X, y, epochs=5)
    
    print("🎉 Training completed! Now run: streamlit run app.py")