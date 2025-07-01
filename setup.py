import os
import argparse
import shutil
import urllib.request
import zipfile
import random
from pathlib import Path
import numpy as np
from PIL import Image

def create_project_structure():
    """Create the necessary project directories"""
    directories = [
        'data',
        'models',
        'static',
        'static/samples',
        'templates',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_sample_data(target_dir='data', sample_count=20):
    """
    Download a subset of the breast cancer histopathology dataset
    For this setup script, we'll use a small publicly available subset
    
    In a real application, you would download the full dataset from Kaggle:
    https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    """
    # URL for a small subset of the data (example purposes only)
    sample_data_url = "https://github.com/example/breast-histopathology-samples/archive/main.zip"
    
    try:
        # Create a placeholder message for the actual implementation
        print(f"In a real implementation, this would download the dataset from Kaggle.")
        print(f"For this demo, we'll create some placeholder sample images.")
        
        # Create some sample images for testing
        create_sample_test_images(sample_count)
        
        return True
    except Exception as e:
        print(f"Error downloading sample data: {e}")
        return False

def create_sample_test_images(count=20):
    """Create synthetic sample images for testing"""
    samples_dir = 'static/samples'
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create random test images
    print(f"Generating {count} sample test images...")
    
    # Helper function to generate a sample cell image
    def generate_sample_image(is_idc, size=(50, 50)):
        # Create a blank image
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Background color (light pink/white for tissue background)
        img[:, :] = [240, 230, 230]
        
        # Number of cells to draw
        num_cells = random.randint(5, 15)
        
        # Draw cells
        for _ in range(num_cells):
            # Cell position
            cx = random.randint(5, size[0] - 5)
            cy = random.randint(5, size[1] - 5)
            
            # Cell size
            cell_size = random.randint(3, 8)
            
            # Cell color (purplish for normal, darker/irregular for IDC)
            if is_idc:
                # IDC cells are darker, more irregular
                color = [random.randint(50, 100), random.randint(10, 30), random.randint(80, 120)]
                # Draw irregular shape
                for i in range(cell_size):
                    # Irregular cell shape
                    offset_x = random.randint(-2, 2)
                    offset_y = random.randint(-2, 2)
                    img[max(0, cy-i+offset_y):min(size[1], cy+i+offset_y), 
                        max(0, cx-i+offset_x):min(size[0], cx+i+offset_x)] = color
            else:
                # Normal cells are more regular and lighter purple
                color = [random.randint(150, 180), random.randint(100, 150), random.randint(170, 210)]
                # Draw circular cell
                for i in range(cell_size):
                    img[max(0, cy-i):min(size[1], cy+i), 
                        max(0, cx-i):min(size[0], cx+i)] = color
        
        return img
    
    # Generate positive and negative samples
    for i in range(count):
        # Decide if this is an IDC or non-IDC sample
        is_idc = i < count // 2
        
        # Generate the image
        img_array = generate_sample_image(is_idc, size=(100, 100))
        img = Image.fromarray(img_array)
        
        # Save the image
        prefix = "idc" if is_idc else "normal"
        img.save(f"{samples_dir}/{prefix}_sample_{i+1}.png")
    
    print(f"Created {count} sample images in {samples_dir}")

def copy_templates():
    """Copy HTML templates to the templates directory"""
    # Check if index.html exists in the current directory
    if os.path.exists('index.html'):
        shutil.copy('index.html', 'templates/index.html')
        print("Copied index.html to templates directory")
    else:
        print("Warning: index.html not found in current directory")

def main():
    parser = argparse.ArgumentParser(description='Setup Breast Cancer Detection Project')
    parser.add_argument('--download', action='store_true', help='Download sample dataset')
    parser.add_argument('--samples', type=int, default=20, help='Number of sample images to generate')
    
    args = parser.parse_args()
    
    print("Setting up Breast Cancer Detection Project...")
    
    # Create project structure
    create_project_structure()
    
    # Copy templates
    copy_templates()
    
    # Download sample data if requested
    if args.download:
        download_sample_data(sample_count=args.samples)
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Train your model with: python app_train.py")
    print("2. Run the Flask app with: python app.py")

if __name__ == "__main__":
    main()