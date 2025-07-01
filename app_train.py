import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import joblib

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Detection - Model Training",
    page_icon="🔬",
    layout="wide"
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess the dataset
data_path=data
def load_dataset(data_path):
    image_size = 50
    images = []
    labels = []
    patient_ids = []
    
    # Get all patient folders
    patient_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # Progress bar
    progress_bar = st.progress(0)
    
    for i, patient_id in enumerate(patient_folders):
        patient_path = os.path.join(data_path, patient_id)
        
        # Get the 0 and 1 folders (if they exist)
        for class_id in ['0', '1']:
            class_path = os.path.join(patient_path, class_id)
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
                
                # Limit the number of images per class and patient for development
                sample_size = min(100, len(image_files))
                sampled_files = random.sample(image_files, sample_size)
                
                for img_file in sampled_files:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path).resize((image_size, image_size))
                        img_array = np.array(img) / 255.0  # Normalize
                        images.append(img_array)
                        labels.append(int(class_id))
                        patient_ids.append(patient_id)
                    except Exception as e:
                        st.warning(f"Error loading image {img_path}: {e}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(patient_folders))
    
    return np.array(images), np.array(labels), np.array(patient_ids)

# Function to create and train the model
def create_model(input_shape):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Function to plot metrics
def plot_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    return fig

# Main app
def main():
    st.title("Breast Cancer Detection - Model Training")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Dataset path input
    dataset_path = st.sidebar.text_input(
        "Enter dataset path (e.g., 'data/breast-histopathology-images/')",
        "data/breast-histopathology-images/"
    )
    
    # Training parameters
    st.sidebar.header("Training Parameters")
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32, 8)
    epochs = st.sidebar.slider("Epochs", 5, 50, 15, 5)
    validation_split = st.sidebar.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
    
    # Data augmentation
    st.sidebar.header("Data Augmentation")
    use_augmentation = st.sidebar.checkbox("Use Data Augmentation", True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Model Training", "Evaluation", "Export"])
    
    # Tab 1: Data Exploration
    with tab1:
        st.header("Data Exploration")
        
        # Button to load dataset
        if st.button("Load and Explore Dataset"):
            if not os.path.exists(dataset_path):
                st.error(f"Dataset path not found: {dataset_path}")
            else:
                st.write("Loading dataset... This may take a while.")
                start_time = time.time()
                X, y, patient_ids = load_dataset(dataset_path)
                
                # Save the shapes and time taken
                st.session_state['dataset_loaded'] = True
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['patient_ids'] = patient_ids
                st.session_state['load_time'] = time.time() - start_time
                
                # Display basic info
                st.write(f"Dataset loaded in {st.session_state['load_time']:.2f} seconds.")
                st.write(f"Number of images: {len(X)}")
                st.write(f"Image shape: {X[0].shape}")
                st.write(f"Number of positive samples (IDC): {sum(y)}")
                st.write(f"Number of negative samples (non-IDC): {len(y) - sum(y)}")
                st.write(f"Number of unique patients: {len(set(patient_ids))}")
                
                # Display class distribution
                st.subheader("Class Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x=y, ax=ax)
                ax.set_xlabel("Class (0: non-IDC, 1: IDC)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
                # Display sample images
                st.subheader("Sample Images")
                col1, col2 = st.columns(2)
                
                # Sample positive images
                positive_indices = np.where(y == 1)[0]
                positive_samples = random.sample(list(positive_indices), min(5, len(positive_indices)))
                
                # Sample negative images
                negative_indices = np.where(y == 0)[0]
                negative_samples = random.sample(list(negative_indices), min(5, len(negative_indices)))
                
                with col1:
                    st.write("Positive samples (IDC)")
                    for idx in positive_samples:
                        st.image(X[idx], caption=f"Patient ID: {patient_ids[idx]}", use_column_width=True)
                
                with col2:
                    st.write("Negative samples (non-IDC)")
                    for idx in negative_samples:
                        st.image(X[idx], caption=f"Patient ID: {patient_ids[idx]}", use_column_width=True)
        
        # If dataset is already loaded
        elif 'dataset_loaded' in st.session_state and st.session_state['dataset_loaded']:
            X = st.session_state['X']
            y = st.session_state['y']
            patient_ids = st.session_state['patient_ids']
            
            st.write(f"Dataset already loaded in {st.session_state['load_time']:.2f} seconds.")
            st.write(f"Number of images: {len(X)}")
            st.write(f"Image shape: {X[0].shape}")
            st.write(f"Number of positive samples (IDC): {sum(y)}")
            st.write(f"Number of negative samples (non-IDC): {len(y) - sum(y)}")
            st.write(f"Number of unique patients: {len(set(patient_ids))}")
            
            # Display class distribution
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=y, ax=ax)
            ax.set_xlabel("Class (0: non-IDC, 1: IDC)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
    
    # Tab 2: Model Training
    with tab2:
        st.header("Model Training")
        
        # Check if dataset is loaded
        if 'dataset_loaded' not in st.session_state or not st.session_state['dataset_loaded']:
            st.warning("Please load the dataset first in the Data Exploration tab.")
        else:
            X = st.session_state['X']
            y = st.session_state['y']
            patient_ids = st.session_state['patient_ids']
            
            if st.button("Train Model"):
                st.write("Splitting data...")
                
                # Split by patient to avoid data leakage
                unique_patients = list(set(patient_ids))
                train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
                
                # Create train and test masks
                train_mask = np.isin(patient_ids, train_patients)
                test_mask = np.isin(patient_ids, test_patients)
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                # Further split training data into train and validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
                )
                
                st.write(f"Training set: {X_train.shape[0]} images")
                st.write(f"Validation set: {X_val.shape[0]} images")
                st.write(f"Test set: {X_test.shape[0]} images")
                
                # Save the splits in session state
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_val'] = X_val
                st.session_state['y_val'] = y_val
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                # Data augmentation
                if use_augmentation:
                    st.write("Setting up data augmentation...")
                    train_datagen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest'
                    )
                    
                    val_datagen = ImageDataGenerator()
                    
                    # Create generators
                    train_generator = train_datagen.flow(
                        X_train, y_train,
                        batch_size=batch_size
                    )
                    
                    val_generator = val_datagen.flow(
                        X_val, y_val,
                        batch_size=batch_size
                    )
                    
                    # Show augmented images
                    st.subheader("Sample Augmented Images")
                    augmented_images, _ = next(train_generator)
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        if i < len(augmented_images):
                            col.image(augmented_images[i], use_column_width=True)
                
                # Create and compile model
                st.write("Creating model...")
                input_shape = X_train[0].shape
                model = create_model(input_shape)
                
                # Model summary
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text("\n".join(model_summary))
                
                # Callbacks
                checkpoint = ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    mode='min',
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6,
                    mode='min',
                    verbose=1
                )
                
                callbacks = [checkpoint, early_stopping, reduce_lr]
                
                # Start training
                st.write("Starting training...")
                training_progress = st.progress(0)
                status_text = st.empty()
                
                # Create a custom callback to update the progress bar
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        training_progress.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")
                
                callbacks.append(StreamlitCallback())
                
                # Train the model
                start_time = time.time()
                
                if use_augmentation:
                    history = model.fit(
                        train_generator,
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(X_val) // batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )
                else:
                    history = model.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=0
                    )
                
                training_time = time.time() - start_time
                st.write(f"Training completed in {training_time:.2f} seconds.")
                
                # Save model and history in session state
                st.session_state['model'] = model
                st.session_state['history'] = history
                st.session_state['model_trained'] = True
                
                # Plot metrics
                st.subheader("Training Metrics")
                metrics_fig = plot_metrics(history)
                st.pyplot(metrics_fig)
    
    # Tab 3: Evaluation
    with tab3:
        st.header("Model Evaluation")
        
        if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
            st.warning("Please train the model first in the Model Training tab.")
        else:
            model = st.session_state['model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            if st.button("Evaluate Model"):
                # Evaluate on test set
                st.write("Evaluating model on test set...")
                test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
                
                st.write(f"Test Loss: {test_loss:.4f}")
                st.write(f"Test Accuracy: {test_acc:.4f}")
                st.write(f"Test AUC: {test_auc:.4f}")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = tf.math.confusion_matrix(y_test, y_pred_binary).numpy()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['non-IDC', 'IDC'])
                ax.set_yticklabels(['non-IDC', 'IDC'])
                st.pyplot(fig)
                
                # Sample incorrect predictions
                incorrect_indices = np.where(y_pred_binary != y_test)[0]
                
                if len(incorrect_indices) > 0:
                    st.subheader("Sample Misclassifications")
                    
                    # Sample a few misclassifications
                    sample_size = min(6, len(incorrect_indices))
                    sampled_indices = random.sample(list(incorrect_indices), sample_size)
                    
                    cols = st.columns(3)
                    for i, idx in enumerate(sampled_indices):
                        col = cols[i % 3]
                        true_label = "IDC" if y_test[idx] == 1 else "non-IDC"
                        pred_label = "IDC" if y_pred_binary[idx] == 1 else "non-IDC"
                        confidence = y_pred[idx][0] if y_pred_binary[idx] == 1 else 1 - y_pred[idx][0]
                        
                        col.image(X_test[idx], use_column_width=True)
                        col.write(f"True: {true_label}, Pred: {pred_label}")
                        col.write(f"Confidence: {confidence:.2f}")
    
    # Tab 4: Export
    with tab4:
        st.header("Export Model")
        
        if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
            st.warning("Please train the model first in the Model Training tab.")
        else:
            model = st.session_state['model']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Model (H5)"):
                    model.save('breast_cancer_model.h5')
                    st.success("Model saved as 'breast_cancer_model.h5'")
            
            with col2:
                if st.button("Save Model (TFLite)"):
                    # Convert to TFLite
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()
                    
                    # Save TFLite model
                    with open('breast_cancer_model.tflite', 'wb') as f:
                        f.write(tflite_model)
                    
                    st.success("Model saved as 'breast_cancer_model.tflite'")
            
            # Save additional information
            if st.button("Save Preprocessing Info"):
                preprocess_info = {
                    'image_size': 50,  # The size used during training
                    'normalization_factor': 255.0  # Normalization factor used
                }
                
                joblib.dump(preprocess_info, 'preprocess_info.joblib')
                st.success("Preprocessing information saved as 'preprocess_info.joblib'")

if __name__ == "__main__":
    main()