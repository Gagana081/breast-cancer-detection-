import numpy as np
import tensorflow as tf

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM with a trained model and target layer
        
        Args:
            model: TensorFlow Keras model
            layer_name: Name of the target convolutional layer. If None, use the last conv layer
        """
        self.model = model
        
        # If layer name is not specified, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        # Store the target layer
        self.layer_name = layer_name
        self.grad_model = self._build_gradient_model()
    
    def _build_gradient_model(self):
        """Build a gradient model for GradCAM"""
        # Get the target layer
        grad_layer = self.model.get_layer(self.layer_name)
        
        # Create a new model that outputs both the final prediction and the activations
        inputs = self.model.inputs
        outputs = [self.model.output, grad_layer.output]
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    def compute_heatmap(self, img_array, pred_class_idx=None, eps=1e-8):
        """
        Compute GradCAM heatmap for the input image
        
        Args:
            img_array: Input image as numpy array (including batch dimension)
            pred_class_idx: Class index for which to generate the heatmap
                            If None, use the predicted class
            eps: Small epsilon value to avoid division by zero
            
        Returns:
            Normalized heatmap as numpy array
        """
        # Get predictions and feature map activations
        with tf.GradientTape() as tape:
            # Cast the image to float32
            img_tensor = tf.cast(img_array, tf.float32)
            
            # Watch the input tensor
            tape.watch(img_tensor)
            
            # Get predictions and activations
            preds, activations = self.grad_model(img_tensor)
            
            # If pred_class_idx is not provided, use the predicted class
            if pred_class_idx is None:
                pred_class_idx = tf.argmax(preds[0])
            
            # Get the prediction for the specified class
            class_channel = preds[:, pred_class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, activations)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute the weighted combination of feature maps
        activations = activations.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        # Apply weights to feature maps
        for i in range(pooled_grads.shape[-1]):
            activations[:, :, i] *= pooled_grads[i]
        
        # Average along the feature map channels
        heatmap = np.mean(activations, axis=-1)
        
        # ReLU to keep only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        
        return heatmap