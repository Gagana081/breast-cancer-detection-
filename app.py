from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'model/breast_cancer_model.h5'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create a static/uploads directory for serving images
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

# Global variable for the model
model = None
target_size = 50  # Ensure this matches your training input size

def load_breast_cancer_model():
    """Load the pre-trained breast cancer detection model"""
    global model
    if model is None:
        try:
            model = load_model(app.config['MODEL_PATH'])
            # Initialize the model by making a prediction on a dummy input
            dummy_input = np.zeros((1, target_size, target_size, 3))
            _ = model.predict(dummy_input)
            print("Model loaded and initialized successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

def preprocess_image(image, target_size=50):
    """Preprocess the image for model prediction"""
    # Resize the image
    image = image.resize((target_size, target_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Ensure the image has 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def generate_gradcam(model, img_array, target_size=50):
    """Generate Grad-CAM visualization to highlight regions of interest"""
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if not last_conv_layer:
            print("No convolutional layer found in the model")
            return None
        
        # Create a model that goes from the input to the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[
                model.get_layer(last_conv_layer).output,
                model.output
            ]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Cast the image tensor to a float-32 data type
            img_array = tf.cast(img_array, tf.float32)
            
            # Compute the gradients of the target class with respect to the last conv layer
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]  # Class 0 or 1 for binary prediction
        
        # Extract the gradients from the targeted class
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the batch dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize the heatmap to match the original image size
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        # Upscale heatmap to original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((target_size, target_size))
        heatmap = np.array(heatmap)
        
        # Apply colormap to heatmap
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)
        heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # Superimpose the heatmap on original image
        original_img = np.uint8(255 * img_array[0])
        superimposed_img = heatmap_colored * 0.4 + original_img * 0.6
        superimposed_img = np.uint8(superimposed_img)
        
        # Create the figure with both images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(superimposed_img)
        ax2.set_title('Grad-CAM Activation')
        ax2.axis('off')
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Encode the buffer to base64 for HTML embedding
        data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return data
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    # Pre-load the model at startup
    load_breast_cancer_model()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction"""
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    
    # Load model if not already loaded
    if not load_breast_cancer_model():
        return render_template('index.html', error="Error loading model. Please check if model file exists.")
    
    try:
        # Read and preprocess the image
        image = Image.open(file.stream)
        img_array = preprocess_image(image, target_size)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Generate class label
        class_label = "Invasive Ductal Carcinoma (IDC) Detected" if prediction > 0.5 else "No Cancer Detected"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        # Generate Grad-CAM visualization
        gradcam_img = generate_gradcam(model, img_array, target_size)
        
        # Save the uploaded image to both the uploads and static/uploads directories
        filename = f"image_{int(time.time())}.png"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        static_path = os.path.join('static', 'uploads', filename)
        
        image.save(upload_path)
        image.save(static_path)
        
        return render_template(
            'result.html',
            prediction=prediction,
            class_label=class_label,
            confidence=float(confidence) * 100,
            image_file=filename,
            gradcam=gradcam_img
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    # Load the model at startup
    load_breast_cancer_model()
    app.run(debug=True)