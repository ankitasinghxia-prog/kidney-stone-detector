import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Kidney Stone Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KidneyStoneApp:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)  
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model('kidney_stone_model.h5')
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Model not found: {e}")
            st.sidebar.info("Running in demo mode with sample predictions")
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for prediction - FIXED to match training"""
        
        img = np.array(image)
        
        
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image):
        """Make prediction on the image"""
        if self.model is None:
            
            demo_confidence = np.random.uniform(0.1, 0.9)
            if demo_confidence > 0.5:
                return demo_confidence, "Stone"
            else:
                return demo_confidence, "Normal"
        
        try:
            preprocessed_img = self.preprocess_image(image)
            prediction = self.model.predict(preprocessed_img, verbose=0)[0][0]
            
            return prediction, "Stone" if prediction > 0.5 else "Normal"
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0.5, "Error"
    
    def run(self):
        """Run the Streamlit application"""
        
        st.sidebar.title("🏥 Kidney Stone Detection")
        st.sidebar.markdown("---")
        
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Kidney Ultrasound Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an ultrasound image of the kidney"
        )
        
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Demo")
        demo_mode = st.sidebar.checkbox("Use Demo Mode", value=True)
        
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "This AI tool helps detect kidney stones in ultrasound images. "
            "It is for educational and demonstration purposes only."
        )
        
        st.sidebar.warning(
            "⚠️ **Medical Disclaimer:**\n"
            "This is not a medical device. Always consult healthcare professionals "
            "for medical diagnosis and treatment."
        )
        
       
        st.title("🧊 Kidney Stone Detection System")
        st.markdown("Upload an ultrasound image to detect the presence of kidney stones.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Input Image")
            
            if uploaded_file is not None:
                
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Ultrasound Image", use_column_width=True)
                
                
                with st.spinner("🔍 Analyzing image for kidney stones..."):
                    confidence, result = self.predict(image)
                
                
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    
                    st.metric("Confidence Level", f"{confidence:.2%}")
                    
                    
                    if result == "Stone":
                        st.error(f"🔴 **KIDNEY STONE DETECTED**")
                        st.warning("""
                        **Potential kidney stone detected.**
                        
                        Please consult a healthcare professional for:
                        - Proper medical diagnosis
                        - Treatment options
                        - Further testing if needed
                        """)
                    elif result == "Normal":
                        st.success(f"🟢 **NORMAL KIDNEY**")
                        st.info("""
                        **No kidney stones detected in the analysis.**
                        
                        The kidney appears normal in this ultrasound.
                        """)
                    else:
                        st.warning("⚠️ **ANALYSIS INCOMPLETE**")
                    
                    
                    st.progress(float(confidence))
                    st.caption(f"Model confidence: {confidence:.2%}")
                    
                    
                    st.markdown("---")
                    st.subheader("📋 Interpretation Guide")
                    
                    if result == "Stone":
                        st.markdown("""
                        **What this means:**
                        - The AI detected features consistent with kidney stones
                        - Stones appear as bright white spots with shadowing
                        - Further medical evaluation recommended
                        """)
                    else:
                        st.markdown("""
                        **What this means:**
                        - No stone-like features detected
                        - Kidney structure appears normal
                        - Regular checkups are still recommended
                        """)
            
            else:
                
                st.info("👆 Please upload a kidney ultrasound image using the file uploader in the sidebar")
                
                
                st.markdown("---")
                st.subheader("📸 Sample Ultrasound Images")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Normal Kidney**")
                    st.image("https://via.placeholder.com/300x200/4A90E2/FFFFFF?text=Normal+Kidney+Ultrasound", 
                            caption="Clear kidney structure, no stones")
                
                with col2:
                    st.markdown("**Kidney with Stone**")
                    st.image("https://via.placeholder.com/300x200/E74C3C/FFFFFF?text=Stone+Detected+Ultrasound", 
                            caption="Bright echo with acoustic shadowing")
                
                st.markdown("""
                **What to look for in kidney ultrasound images:**
                - **Normal**: Clear kidney structure, no bright echoes
                - **Stone**: Hyperechoic foci with acoustic shadowing
                - **Swelling**: Hydronephrosis (enlarged kidney)
                """)
        
        
        with st.expander("🔧 Technical Information"):
            st.markdown("""
            **Model Information:**
            - **Architecture**: Convolutional Neural Network (CNN)
            - **Input Size**: 128x128 pixels
            - **Training**: Kidney ultrasound images
            - **Output**: Binary classification (Stone/Normal)
            
            **How it works:**
            1. Image resized to 128x128 pixels
            2. Converted to RGB format
            3. Normalized pixel values
            4. Analyzed by trained AI model
            5. Confidence score generated
            
            **Note**: This is a demonstration system. For actual medical diagnosis, 
            always consult qualified healthcare professionals.
            """)
            
            if self.model is not None:
                st.success("✅ AI Model: Active and Loaded")
            else:
                st.info("🟡 Demo Mode: Using simulated predictions")


if __name__ == "__main__":
    app = KidneyStoneApp()
    app.run()