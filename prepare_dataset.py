import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

class DatasetValidator:
    """Validate dataset structure and contents."""
    
    @staticmethod
    def validate_image(img_path):
        try:
            with Image.open(img_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    return False, f"Invalid image mode: {img.mode}"
                if min(img.size) < 32:
                    return False, f"Image too small: {img.size}"
                return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_dataset_structure(data_dir, valid_fruits, valid_ripeness):
        issues = []
        for fruit in valid_fruits:
            fruit_dir = os.path.join(data_dir, fruit)
            if not os.path.exists(fruit_dir):
                issues.append(f"Missing fruit directory: {fruit}")
                continue
                
            for ripeness in valid_ripeness:
                ripeness_dir = os.path.join(fruit_dir, ripeness)
                if not os.path.exists(ripeness_dir):
                    issues.append(f"Missing ripeness directory: {fruit}/{ripeness}")
                    
        return issues

class ImagePreprocessor:
    """Handle image preprocessing operations."""
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def enhance_fruit_features(self, img):
        """Enhanced feature extraction for fruit images."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Enhance color channels
        a = cv2.equalizeHist(a)
        b = cv2.equalizeHist(b)
        
        # Merge channel
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply careful denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def normalize_image(self, img):
        """Normalize image with advanced techniques."""
        # Convert to float32
        img = img.astype(np.float32)
        
        # Channel-wise normalization
        for i in range(3):
            mean = np.mean(img[:,:,i])
            std = np.std(img[:,:,i])
            img[:,:,i] = (img[:,:,i] - mean) / (std + 1e-7)
        
        return img
    
    def process_image(self, img_path):
        """Complete image processing pipeline."""
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = np.array(img)
                img = cv2.resize(img, self.img_size)
                img = self.enhance_fruit_features(img)
                img = self.normalize_image(img)
                return img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

class DataAugmenter:
    """Handle data augmentation with advanced techniques."""
    
    def __init__(self):
        self.augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),  # Replace deprecated Flip
            A.VerticalFlip(p=0.5),    # Added vertical flip
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Replace IAAAdditiveGaussianNoise
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.2, 
                rotate_limit=45, 
                p=0.2
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),  # Replace IAAPiecewiseAffine
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(p=0.2),
                A.Emboss(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ], bbox_params=A.BboxParams(format='coco', label_fields=[]))
    
    def create_augmented_data(self, X, y_fruit, y_ripeness, augmentation_factor=3):
        """Create augmented versions of the training data."""
        print(f"Starting advanced data augmentation with {len(X)} original images")
        augmented_X = []
        augmented_y_fruit = []
        augmented_y_ripeness = []
        
        # Add original images
        augmented_X.extend(X)
        augmented_y_fruit.extend(y_fruit)
        augmented_y_ripeness.extend(y_ripeness)
        
        # Create augmented versions
        for i in range(len(X)):
            for _ in range(augmentation_factor):
                augmented = self.augmentation(image=X[i])['image']
                augmented_X.append(augmented)
                augmented_y_fruit.append(y_fruit[i])
                augmented_y_ripeness.append(y_ripeness[i])
        
        return np.array(augmented_X), np.array(augmented_y_fruit), np.array(augmented_y_ripeness)

def load_data(data_dir, img_size=(128, 128)):
    """Enhanced data loading with validation and preprocessing."""
    X = []
    y_fruit = []
    y_ripeness = []
    
    valid_fruits = ['banana', 'apple']
    valid_ripeness = ['unripe', 'ripe', 'overripe']
    
    # Initialize components
    validator = DatasetValidator()
    preprocessor = ImagePreprocessor(img_size)
    
    # Validate dataset structure
    issues = validator.validate_dataset_structure(data_dir, valid_fruits, valid_ripeness)
    if issues:
        print("Dataset structure issues found:")
        for issue in issues:
            print(f"- {issue}")
        if len(issues) > len(valid_fruits):
            raise ValueError("Too many dataset structure issues to proceed")
    
    # Process each fruit type
    available_categories = {}
    
    for fruit_idx, fruit_name in enumerate(valid_fruits):
        fruit_dir = os.path.join(data_dir, fruit_name)
        if not os.path.exists(fruit_dir):
            continue
            
        available_categories[fruit_name] = []
        print(f"\nProcessing {fruit_name}...")
        
        for ripeness_idx, ripeness in enumerate(valid_ripeness):
            ripeness_dir = os.path.join(fruit_dir, ripeness)
            if not os.path.exists(ripeness_dir):
                continue
                
            image_files = [f for f in os.listdir(ripeness_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                continue
                
            print(f"Processing {len(image_files)} images in {fruit_name}/{ripeness}")
            available_categories[fruit_name].append(ripeness)
            
            # Process images in parallel
            with ThreadPoolExecutor() as executor:
                image_paths = [os.path.join(ripeness_dir, img) for img in image_files]
                processed_images = list(executor.map(preprocessor.process_image, image_paths))
                
                for img in processed_images:
                    if img is not None:
                        X.append(img)
                        y_fruit.append(fruit_idx)
                        y_ripeness.append(valid_ripeness.index(ripeness))
    
    if not X:
        raise ValueError("No valid images were loaded!")
    
    return np.array(X), np.array(y_fruit), np.array(y_ripeness), valid_fruits, available_categories

def main():
    # Configuration
    config = {
        'data_dir': r"D:\AI Hackathon\ALML MultiFruit Ripeness detection Project\dataset",
        'img_size': (128, 128),
        'augmentation_factor': 3,
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Create output directory
    output_dir = Path('processed_dataset')
    output_dir.mkdir(exist_ok=True)
    
    print("Starting dataset preparation...")
    print(f"Looking for data in: {config['data_dir']}")
    
    try:
        # Load and preprocess data
        X, y_fruit, y_ripeness, fruit_labels, available_categories = load_data(
            config['data_dir'], 
            config['img_size']
        )
        
        # Perform augmentation
        augmenter = DataAugmenter()
        X_augmented, y_fruit_augmented, y_ripeness_augmented = augmenter.create_augmented_data(
            X, y_fruit, y_ripeness,
            config['augmentation_factor']
        )
        
        # Convert labels to categorical
        num_fruits = len(fruit_labels)
        num_ripeness = len(set(y_ripeness))
        
        y_fruit_categorical = tf.keras.utils.to_categorical(y_fruit_augmented, num_fruits)
        y_ripeness_categorical = tf.keras.utils.to_categorical(y_ripeness_augmented, num_ripeness)
        
        # Split dataset
        X_train, X_test, y_fruit_train, y_fruit_test, y_ripeness_train, y_ripeness_test = train_test_split(
            X_augmented, y_fruit_categorical, y_ripeness_categorical,
            test_size=config['test_size'], 
            random_state=config['random_state']
        )
        
        # Save processed data
        print("\nSaving processed data...")
        for name, arr in {
            'X_train': X_train,
            'X_test': X_test,
            'y_fruit_train': y_fruit_train,
            'y_fruit_test': y_fruit_test,
            'y_ripeness_train': y_ripeness_train,
            'y_ripeness_test': y_ripeness_test,
        }.items():
            np.save(output_dir / f'{name}.npy', arr)
        
        # Save metadata
        metadata = {
            'fruit_labels': fruit_labels,
            'available_categories': available_categories,
            'config': config,
            'dataset_stats': {
                'total_original_images': len(X),
                'total_augmented_images': len(X_augmented),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nDataset preparation completed successfully!")
        print(f"\nDataset statistics:")
        print(f"Original images: {len(X)}")
        print(f"Augmented images: {len(X_augmented)}")
        print(f"Training set: {len(X_train)}")
        print(f"Test set: {len(X_test)}")
        
    except Exception as e:
        print(f"\nError during dataset preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
