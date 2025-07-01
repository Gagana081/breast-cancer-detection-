import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import glob

# Set page config
st.set_page_config(page_title="Breast Cancer Model Training", layout="wide")

# Title and description
st.title("Breast Cancer Detection Model Training")
st.markdown("""
This application allows you to train a deep learning model to detect Invasive Ductal Carcinoma (IDC) 
in breast histopathology images. The model is trained on the Breast Histopathology Images dataset from Kaggle.
""")

# Sidebar for training parameters
st.sidebar.header("Training Parameters")

# Dataset parameters
st.sidebar.subheader("Dataset")
dataset_path = st.sidebar.text_input("Dataset Path", "breast-histopathology-images")

# Model parameters
st.sidebar.subheader("Model Configuration")
input_size = st.sidebar.number_input("Image Size", min_value=32, max_value=224, value=50)
batch_size = st.sidebar.number_input("Batch Size", min_value=8, max_value=128, value=32)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=20)
validation_split = st.sidebar.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    format_func=lambda x: f"{x:.4f}",
    value=0.001
)

# Data augmentation parameters
st.sidebar.subheader("Data Augmentation")
use_augmentation = st.sidebar.checkbox("Use Data Augmentation", value=True)
if use_augmentation:
    rotation_range = st.sidebar.slider("Rotation Range", min_value=0, max_value=90, value=20)
    width_shift = st.sidebar.slider("Width Shift", min_value=0.0, max_value=0.3, value=0.1)
    height_shift = st.sidebar.slider("Height Shift", min_value=0.0, max_value=0.3, value=0.1)
    zoom_range = st.sidebar.slider("Zoom Range", min_value=0.0, max_value=0.5, value=0.2)
    horizontal_flip = st.sidebar.checkbox("Horizontal Flip", value=True)
    vertical_flip = st.sidebar.checkbox("Vertical Flip", value=True)

# Early stopping parameters
st.sidebar.subheader("Training Controls")
use_early_stopping = st.sidebar.checkbox("Use Early Stopping", value=True)
if use_early_stopping:
    patience = st.sidebar.number_input("Patience", min_value=1, max_value=20, value=5)

# Function to load and prepare data
def load_data(dataset_path, input_size, batch_size, validation_split):
    st.info("Loading dataset...")
    
    # Collect image paths and labels
    positive_files = []
    negative_files = []
    
    # Get all patient directories
    patient_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Progress bar for data loading
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i, patient_id in enumerate(patient_dirs):
        # Update progress
        progress = (i + 1) / len(patient_dirs)
        progress_bar.progress(progress)
        progress_text.text(f"Processing patient {i+1}/{len(patient_dirs)}: {patient_id}")
        
        # Path to positive and negative samples for this patient
        positive_path = os.path.join(dataset_path, patient_id, "1")
        negative_path = os.path.join(dataset_path, patient_id, "0")
        
        # Collect positive samples
        if os.path.exists(positive_path):
            pos_images = glob.glob(os.path.join(positive_path, "*.png"))
            positive_files.extend(pos_images)
        
        # Collect negative samples
        if os.path.exists(negative_path):
            neg_images = glob.glob(os.path.join(negative_path, "*.png"))
            negative_files.extend(neg_images)
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Create labels
    positive_labels = [1] * len(positive_files)
    negative_labels = [0] * len(negative_files)
    
    # Combine data
    all_files = positive_files + negative_files
    all_labels = positive_labels + negative_labels
    
    # Display dataset statistics
    st.success("Dataset loaded successfully!")
    st.write(f"Total images: {len(all_files)}")
    st.write(f"Positive samples (IDC): {len(positive_files)}")
    st.write(f"Negative samples (non-IDC): {len(negative_files)}")
    
    # Ensure balanced sampling for training
    # We'll use a smaller dataset to keep training time reasonable for demonstration
    max_samples = min(len(positive_files), len(negative_files))
    max_samples = min(max_samples, 10000)  # Limit to 10,000 of each class
    
    # Randomly select equal numbers from positive and negative samples
    np.random.seed(42)
    selected_pos = np.random.choice(positive_files, max_samples, replace=False)
    selected_neg = np.random.choice(negative_files, max_samples, replace=False)
    
    selected_files = np.concatenate([selected_pos, selected_neg])
    selected_labels = np.concatenate([np.ones(max_samples), np.zeros(max_samples)])
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        selected_files, selected_labels, 
        test_size=validation_split, 
        random_state=42,
        stratify=selected_labels
    )
    
    st.write(f"Training samples: {len(X_train)}")
    st.write(f"Validation samples: {len(X_val)}")
    
    # Create a DataFrame for easier handling
    train_df = pd.DataFrame({
        'filename': X_train,
        'class': y_train
    })
    
    val_df = pd.DataFrame({
        'filename': X_val,
        'class': y_val
    })
    
    # Show sample images
    st.subheader("Sample Images")
    col1, col2 = st.columns(2)
    
    # Show a positive sample
    with col1:
        st.write("IDC Positive Sample")
        sample_pos = np.random.choice(selected_pos)
        img = Image.open(sample_pos)
        st.image(img, width=200)
    
    # Show a negative sample
    with col2:
        st.write("IDC Negative Sample")
        sample_neg = np.random.choice(selected_neg)
        img = Image.open(sample_neg)
        st.image(img, width=200)
    
    return train_df, val_df

# Function to create data generators
def create_data_generators(train_df, val_df, input_size, batch_size, use_augmentation):
    # Custom data generator function to load images on the fly
    def generate_data(df, batch_size, input_size, is_training=False):
        n = df.shape[0]
        while True:
            # Shuffle data for each epoch
            df = df.sample(frac=1).reset_index(drop=True)
            
            for i in range(0, n, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Initialize batch arrays
                batch_x = np.zeros((len(batch_df), input_size, input_size, 3))
                batch_y = np.zeros((len(batch_df), 1))
                
                for j, (_, row) in enumerate(batch_df.iterrows()):
                    # Load and preprocess image
                    img = Image.open(row['filename'])
                    img = img.resize((input_size, input_size))
                    img_array = np.array(img) / 255.0  # Normalize
                    
                    # Apply simple augmentation if needed and is training set
                    if is_training and use_augmentation:
                        # Random horizontal flip
                        if horizontal_flip and np.random.random() > 0.5:
                            img_array = np.fliplr(img_array)
                        
                        # Random vertical flip
                        if vertical_flip and np.random.random() > 0.5:
                            img_array = np.flipud(img_array)
                    
                    # Add to batch
                    batch_x[j] = img_array
                    batch_y[j] = row['class']
                
                yield batch_x, batch_y
    
    # Create generators
    train_generator = generate_data(train_df, batch_size, input_size, is_training=True)
    val_generator = generate_data(val_df, batch_size, input_size, is_training=False)
    
    return train_generator, val_generator

# Function to build the model
def build_model(input_size):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(input_size, input_size, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Function to train the model
def train_model(model, train_generator, val_generator, train_df, val_df, epochs, batch_size):
    # Calculate steps per epoch
    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    os.makedirs('model', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'model/breast_cancer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Early stopping if enabled
    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # Progress containers
    history_container = st.empty()
    metrics_container = st.empty()
    progress_bar = st.progress(0)
    
    # Placeholder for history plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    history_plot = st.pyplot(fig)
    
    # History data
    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    # Train model manually to show progress
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        train_loss = 0
        train_acc = 0
        
        for step in range(steps_per_epoch):
            x_batch, y_batch = next(train_generator)
            metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)
            train_loss += metrics['loss']
            train_acc += metrics['accuracy']
            
            # Update progress within epoch
            within_epoch_progress = (step + 1) / steps_per_epoch
            overall_progress = (epoch + within_epoch_progress) / epochs
            progress_bar.progress(overall_progress)
        
        train_loss /= steps_per_epoch
        train_acc /= steps_per_epoch
        
        # Validation
        val_loss = 0
        val_acc = 0
        
        for step in range(validation_steps):
            x_batch, y_batch = next(val_generator)
            metrics = model.test_on_batch(x_batch, y_batch, return_dict=True)
            val_loss += metrics['loss']
            val_acc += metrics['accuracy']
        
        val_loss /= validation_steps
        val_acc /= validation_steps
        
        # Update history
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Display metrics
        epoch_time = time.time() - start_time
        metrics_container.markdown(f"""
        **Epoch {epoch+1}/{epochs}** - {epoch_time:.1f}s
        - Train Loss: {train_loss:.4f}
        - Train Accuracy: {train_acc:.4f}
        - Validation Loss: {val_loss:.4f}
        - Validation Accuracy: {val_acc:.4f}
        """)
        
        # Update plot
        ax[0].clear()
        ax[1].clear()
        
        ax[0].plot(history['accuracy'], label='Train')
        ax[0].plot(history['val_accuracy'], label='Validation')
        ax[0].set_title('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        
        ax[1].plot(history['loss'], label='Train')
        ax[1].plot(history['val_loss'], label='Validation')
        ax[1].set_title('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        
        fig.tight_layout()
        history_plot.pyplot(fig)
        
        # Check for early stopping
        if use_early_stopping and len(history['val_loss']) > patience:
            if history['val_loss'][-patience-1] < min(history['val_loss'][-patience:]): 
                st.warning(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Apply learning rate reduction if necessary
        if len(history['val_loss']) > 3:
            if history['val_loss'][-4] < min(history['val_loss'][-3:]): 
                current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
                new_lr = current_lr * 0.2
                tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                st.info(f"Reducing learning rate to {new_lr:.6f}")
    
    # Save the final model
    model.save('model/breast_cancer_model.h5')
    
    return history

# Function to evaluate the model
def evaluate_model(model, val_generator, val_df, batch_size):
    # Calculate validation steps
    validation_steps = max(1, len(val_df) // batch_size)
    
    # Evaluate model
    st.subheader("Model Evaluation")
    
    # Get predictions
    predictions = []
    ground_truth = []
    
    for _ in range(validation_steps):
        x_batch, y_batch = next(val_generator)
        batch_predictions = model.predict(x_batch)
        predictions.extend(batch_predictions.flatten())
        ground_truth.extend(y_batch.flatten())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Confusion matrix
    binary_predictions = (predictions > 0.5).astype(int)
    cm = confusion_matrix(ground_truth, binary_predictions)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    # Add labels to the plot
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
    
    # Add class labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-IDC', 'IDC'])
    ax.set_yticklabels(['Non-IDC', 'IDC'])
    
    st.pyplot(fig)
    
    # Classification report
    report = classification_report(ground_truth, binary_predictions, target_names=['Non-IDC', 'IDC'])
    st.text("Classification Report")
    st.text(report)

# Main execution
def main():
    # Show model architecture
    st.subheader("Model Architecture")
    if st.button("Preview Model Architecture"):
        model = build_model(input_size)
        model.summary(print_fn=lambda x: st.text(x))
    
    # Start training
    if st.button("Start Training"):
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            st.error(f"Dataset path '{dataset_path}' not found. Please check the path and try again.")
            return
        
        # Load data
        train_df, val_df = load_data(dataset_path, input_size, batch_size, validation_split)
        
        # Create data generators
        train_generator, val_generator = create_data_generators(train_df, val_df, input_size, batch_size, use_augmentation)
        
        # Build model
        st.subheader("Model Training")
        st.info("Building model...")
        model = build_model(input_size)
        
        # Train model
        st.info("Starting training...")
        history = train_model(model, train_generator, val_generator, train_df, val_df, epochs, batch_size)
        
        # Evaluate model
        st.success("Training completed! Evaluating model...")
        evaluate_model(model, val_generator, val_df, batch_size)
        
        # Save model
        st.success("Model saved to 'model/breast_cancer_model.h5'")
        st.info("You can now use this model with the Flask application (app.py)")

if __name__ == "__main__":
    main()