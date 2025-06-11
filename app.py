import streamlit as st
import numpy as np
import pickle
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import io
import base64
import os  # Add this line from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from streamlit_drawable_canvas import st_canvas
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Streamlit Setup & Monkey-Patch
# ---------------------------
st.set_page_config(page_title="Random Forest Ant Detection System", layout="wide")

# Monkey-patch: add missing image_to_url function for drawable canvas
import streamlit.elements.image as st_image
def image_to_url(img, *args, **kwargs):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
st_image.image_to_url = image_to_url

# ---------------------------
# Random Forest Ant Detector Class
# ---------------------------
class RandomForestAntDetector:
    def __init__(self):
        self.model = None
        self.labels = ['background', 'ant']  
        self.image = None  
        self.image_path = None
        self.original_filename = None   
        self.annotations = []  
        self.current_model_path = None
        
        # Feature extraction configuration
        self.patch_size = (32, 32)  # Size of patches to extract
        self.stride = 16  # Increased stride for faster detection (was 8)
        self.scales = [0.75, 1.0, 1.25]  # Reduced scales for faster detection
        
        # Fast detection mode option
        self.fast_mode = True  # Enable faster detection by default
        
        # HOG parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_n_points = 24
        
        # Gabor filter parameters
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Random Forest parameters
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # NMS parameters
        self.nms_threshold = 0.3
        
        # Detection parameters
        self.confidence_threshold = 0.6

    def extract_features(self, patch):
        """
        Extract comprehensive features from an image patch
        
        Args:
            patch: Image patch (numpy array)
            
        Returns:
            Feature vector
        """
        if len(patch.shape) == 3:
            # Convert to grayscale if needed
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray_patch = patch
        
        features = []
        
        # 1. HOG features
        try:
            hog_features = hog(
                gray_patch,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm='L2-Hys',
                feature_vector=True
            )
            features.extend(hog_features)
        except:
            # If HOG fails, add zeros
            features.extend([0] * 324)  # Default HOG feature size
        
        # 2. Local Binary Pattern (LBP) features
        try:
            lbp = local_binary_pattern(
                gray_patch, 
                self.lbp_n_points, 
                self.lbp_radius, 
                method='uniform'
            )
            lbp_hist, _ = np.histogram(
                lbp.ravel(), 
                bins=self.lbp_n_points + 2, 
                range=(0, self.lbp_n_points + 2)
            )
            # Normalize histogram
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist)
        except:
            features.extend([0] * (self.lbp_n_points + 2))
        
        # 3. Gabor filter responses
        for freq in self.gabor_frequencies:
            for angle in self.gabor_angles:
                try:
                    real, _ = gabor(gray_patch, frequency=freq, theta=angle)
                    # Statistical features from Gabor response
                    features.extend([
                        np.mean(real),
                        np.std(real),
                        np.max(real),
                        np.min(real)
                    ])
                except:
                    features.extend([0, 0, 0, 0])
        
        # 4. Basic statistical features
        features.extend([
            np.mean(gray_patch),
            np.std(gray_patch),
            np.max(gray_patch),
            np.min(gray_patch),
            np.median(gray_patch)
        ])
        
        # 5. Edge features
        try:
            edges = cv2.Canny(gray_patch, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
        except:
            features.append(0)
        
        # 6. Texture features (Gray Level Co-occurrence Matrix approximation)
        try:
            # Simple texture measures
            dx = np.diff(gray_patch, axis=1)
            dy = np.diff(gray_patch, axis=0)
            texture_energy = np.mean(dx**2) + np.mean(dy**2)
            features.append(texture_energy)
        except:
            features.append(0)
        
        return np.array(features, dtype=np.float32)

    def extract_patches_and_labels(self, images, annotations_list):
        """
        Extract training patches and labels from annotated images
        
        Args:
            images: List of images
            annotations_list: List of annotation lists for each image
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        X_features = []  # Renamed to avoid conflicts
        y_labels = []    # Renamed to avoid conflicts
        
        for img_idx, (image, annotations) in enumerate(zip(images, annotations_list)):
            st.write(f"Processing image {img_idx + 1}/{len(images)}...")
            
            h, w = image.shape[:2]
            
            # Convert annotations to pixel coordinates
            ant_boxes = []
            for ann in annotations:
                x1, y1, x2, y2 = ann['bbox']
                # Convert from normalized to pixel coordinates
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                ant_boxes.append([x1, y1, x2, y2])
            
            # Extract positive samples (ants)
            positive_count = 0
            for box in ant_boxes:
                x1, y1, x2, y2 = box
                
                # Extract multiple patches around the ant for data augmentation
                ant_center_x = (x1 + x2) // 2
                ant_center_y = (y1 + y2) // 2
                
                # Extract patches at different positions around the ant
                for offset_x in [-4, 0, 4]:
                    for offset_y in [-4, 0, 4]:
                        patch_x = max(0, min(w - self.patch_size[0], 
                                           ant_center_x - self.patch_size[0]//2 + offset_x))
                        patch_y = max(0, min(h - self.patch_size[1], 
                                           ant_center_y - self.patch_size[1]//2 + offset_y))
                        
                        patch = image[patch_y:patch_y + self.patch_size[1], 
                                    patch_x:patch_x + self.patch_size[0]]
                        
                        if patch.shape[:2] == self.patch_size:
                            features = self.extract_features(patch)
                            if len(features) > 0:
                                X_features.append(features)
                                y_labels.append(1)  # Ant
                                positive_count += 1
            
            # Extract negative samples (background)
            # Extract more negative samples to balance the dataset
            negative_target = positive_count * 3  # 3:1 ratio of negative to positive
            negative_count = 0
            
            # Random sampling for negative examples
            attempts = 0
            max_attempts = negative_target * 10
            
            while negative_count < negative_target and attempts < max_attempts:
                attempts += 1
                
                # Random location
                rand_x = np.random.randint(0, max(1, w - self.patch_size[0]))
                rand_y = np.random.randint(0, max(1, h - self.patch_size[1]))
                
                # Check if this patch overlaps with any ant
                patch_center_x = rand_x + self.patch_size[0] // 2
                patch_center_y = rand_y + self.patch_size[1] // 2
                
                is_ant = False
                for box in ant_boxes:
                    bx1, by1, bx2, by2 = box
                    if bx1 <= patch_center_x <= bx2 and by1 <= patch_center_y <= by2:
                        is_ant = True
                        break
                
                if not is_ant:
                    patch = image[rand_y:rand_y + self.patch_size[1], 
                                rand_x:rand_x + self.patch_size[0]]
                    
                    if patch.shape[:2] == self.patch_size:
                        features = self.extract_features(patch)
                        if len(features) > 0:
                            X_features.append(features)
                            y_labels.append(0)  # Background
                            negative_count += 1
            
            st.write(f"  Extracted {positive_count} positive and {negative_count} negative samples")
        
        return np.array(X_features), np.array(y_labels)

    def train_model(self, images, annotations, tune_hyperparameters=True):
        """
        Train the Random Forest model
        
        Args:
            images: List of images
            annotations: List of annotation lists
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Training results
        """
        st.write("Extracting features from training images...")
        X, y = self.extract_patches_and_labels(images, annotations)
        
        if len(X) == 0:
            raise ValueError("No training samples extracted")
        
        st.write(f"Total training samples: {len(X)} (Positive: {np.sum(y)}, Negative: {np.sum(y == 0)})")
        st.write(f"Feature vector size: {X.shape[1]}")
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            st.write("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use a smaller grid for quick tuning if dataset is small
            if len(X) < 1000:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=min(3, len(X)//10), 
                scoring='f1', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            st.write(f"Best parameters: {best_params}")
            st.write(f"Best cross-validation F1 score: {best_score:.3f}")
            
        else:
            # Train with default parameters
            self.model = RandomForestClassifier(**self.rf_params)
            self.model.fit(X, y)
        
        # Training accuracy
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        st.write(f"Training accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def sliding_window_detection(self, image, confidence_threshold=None):
        """
        Perform sliding window detection on an image with progress tracking
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detections with scores
        """
        if self.model is None:
            return []
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        detections = []
        h, w = image.shape[:2]
        
        # Calculate total operations for progress tracking
        total_operations = 0
        for scale in self.scales:
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h >= self.patch_size[1] and new_w >= self.patch_size[0]:
                y_steps = ((new_h - self.patch_size[1]) // self.stride) + 1
                x_steps = ((new_w - self.patch_size[0]) // self.stride) + 1
                total_operations += y_steps * x_steps
        
        if total_operations == 0:
            st.warning("Image too small for detection with current patch size and scales.")
            return []
        
        st.write(f"Processing {total_operations:,} patches across {len(self.scales)} scales...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        operations_completed = 0
        
        # Multi-scale detection
        for scale_idx, scale in enumerate(self.scales):
            st.write(f"Processing scale {scale} ({scale_idx + 1}/{len(self.scales)})...")
            
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h < self.patch_size[1] or new_w < self.patch_size[0]:
                continue
                
            resized_image = cv2.resize(image, (new_w, new_h))
            
            # Calculate step ranges for this scale
            y_range = list(range(0, new_h - self.patch_size[1] + 1, self.stride))
            x_range = list(range(0, new_w - self.patch_size[0] + 1, self.stride))
            
            # Batch process patches for efficiency
            batch_size = 100  # Process patches in batches
            batch_patches = []
            batch_positions = []
            
            # Sliding window
            for y in y_range:
                for x in x_range:
                    # Extract patch
                    patch = resized_image[y:y + self.patch_size[1], 
                                        x:x + self.patch_size[0]]
                    
                    if patch.shape[:2] != self.patch_size:
                        operations_completed += 1
                        continue
                    
                    batch_patches.append(patch)
                    batch_positions.append((x, y, scale))
                    
                    # Process batch when full
                    if len(batch_patches) >= batch_size:
                        detections.extend(self._process_patch_batch(
                            batch_patches, batch_positions, confidence_threshold
                        ))
                        batch_patches = []
                        batch_positions = []
                    
                    operations_completed += 1
                    
                    # Update progress every 100 operations
                    if operations_completed % 100 == 0:
                        progress_bar.progress(min(operations_completed / total_operations, 1.0))
            
            # Process remaining patches in batch
            if batch_patches:
                detections.extend(self._process_patch_batch(
                    batch_patches, batch_positions, confidence_threshold
                ))
        
        progress_bar.progress(1.0)
        st.write(f"Found {len(detections)} potential detections")
        
        return detections
    
    def _process_patch_batch(self, patches, positions, confidence_threshold):
        """
        Process a batch of patches efficiently
        
        Args:
            patches: List of image patches
            positions: List of (x, y, scale) positions
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detections from this batch
        """
        batch_detections = []
        
        # Extract features for all patches in batch
        feature_vectors = []
        valid_indices = []
        
        for i, patch in enumerate(patches):
            try:
                features = self.extract_features(patch)
                if len(features) > 0:
                    feature_vectors.append(features)
                    valid_indices.append(i)
            except:
                continue
        
        if not feature_vectors:
            return batch_detections
        
        # Batch predict for efficiency
        try:
            # Convert to numpy array for batch prediction
            feature_matrix = np.array(feature_vectors)
            
            # Predict probabilities for entire batch
            probabilities = self.model.predict_proba(feature_matrix)
            
            # Process results
            for prob_idx, prob in enumerate(probabilities):
                original_idx = valid_indices[prob_idx]
                x, y, scale = positions[original_idx]
                
                ant_prob = prob[1]  # Probability of being an ant
                
                if ant_prob >= confidence_threshold:
                    # Convert coordinates back to original scale
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + self.patch_size[0]) / scale)
                    y2 = int((y + self.patch_size[1]) / scale)
                    
                    batch_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': ant_prob,
                        'scale': scale
                    })
        except Exception as e:
            # Fallback to individual processing if batch fails
            st.warning(f"Batch processing failed, using individual processing: {e}")
            for i, patch in enumerate(patches):
                if i not in valid_indices:
                    continue
                    
                try:
                    features = self.extract_features(patch)
                    if len(features) == 0:
                        continue
                    
                    probabilities = self.model.predict_proba([features])[0]
                    ant_prob = probabilities[1]
                    
                    if ant_prob >= confidence_threshold:
                        x, y, scale = positions[i]
                        x1 = int(x / scale)
                        y1 = int(y / scale)
                        x2 = int((x + self.patch_size[0]) / scale)
                        y2 = int((y + self.patch_size[1]) / scale)
                        
                        batch_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': ant_prob,
                            'scale': scale
                        })
                except:
                    continue
        
        return batch_detections

    def non_max_suppression(self, detections, threshold=None):
        """
        Apply non-maximum suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            threshold: IoU threshold for suppression
            
        Returns:
            Filtered detections
        """
        if threshold is None:
            threshold = self.nms_threshold
        
        if not detections:
            return []
        
        # Convert to format for NMS
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['score'] for det in detections])
        
        # Get indices sorted by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = ((remaining_boxes[:, 2] - remaining_boxes[:, 0]) * 
                             (remaining_boxes[:, 3] - remaining_boxes[:, 1]))
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-7)
            
            # Keep boxes with IoU less than threshold
            indices = indices[1:][iou < threshold]
        
        return [detections[i] for i in keep]

    def predict(self, image, confidence_threshold=None, nms_threshold=None):
        """
        Predict ant locations in an image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of (boxes, scores, all_detections)
        """
        if self.model is None:
            return [], [], []
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        
        # Store original image for later use
        self.image = image
        
        # Adjust parameters for fast mode
        original_stride = self.stride
        original_scales = self.scales.copy()
        
        if hasattr(self, 'fast_mode') and self.fast_mode:
            # Use faster settings
            self.stride = max(16, self.stride)  # Larger stride
            self.scales = [1.0]  # Single scale only
            st.info("Fast mode enabled: Using larger stride and single scale for faster detection")
        
        try:
            # Perform sliding window detection
            detections = self.sliding_window_detection(image, confidence_threshold)
            
            if not detections:
                return [], [], []
            
            # Apply NMS
            st.write("Applying Non-Maximum Suppression...")
            filtered_detections = self.non_max_suppression(detections, nms_threshold)
            
            # Extract boxes and scores
            boxes = [det['bbox'] for det in filtered_detections]
            scores = [det['score'] for det in filtered_detections]
            
            return boxes, scores, detections
        
        finally:
            # Restore original parameters
            self.stride = original_stride
            self.scales = original_scales

    def save_model_to_bytes(self):
        """Save the trained model and configuration to bytes for download"""
        if self.model is not None:
            # Create a BytesIO buffer for the model
            model_buffer = io.BytesIO()
            
            # Create a dictionary with model and config
            save_dict = {
                'model': self.model,
                'config': {
                    'labels': self.labels,
                    'patch_size': self.patch_size,
                    'stride': self.stride,
                    'scales': self.scales,
                    'hog_orientations': self.hog_orientations,
                    'hog_pixels_per_cell': self.hog_pixels_per_cell,
                    'hog_cells_per_block': self.hog_cells_per_block,
                    'lbp_radius': self.lbp_radius,
                    'lbp_n_points': self.lbp_n_points,
                    'gabor_frequencies': self.gabor_frequencies,
                    'gabor_angles': [float(a) for a in self.gabor_angles],
                    'rf_params': self.rf_params,
                    'nms_threshold': self.nms_threshold,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            # Pickle the dictionary
            pickle.dump(save_dict, model_buffer)
            model_buffer.seek(0)
            
            return model_buffer
        return None

    def load_model_from_bytes(self, model_bytes):
        """Load a trained model and configuration from bytes"""
        try:
            # Load the dictionary
            save_dict = pickle.load(io.BytesIO(model_bytes))
            
            # Load the model
            self.model = save_dict['model']
            
            # Load configuration
            config = save_dict.get('config', {})
            
            self.labels = config.get('labels', self.labels)
            self.patch_size = tuple(config.get('patch_size', self.patch_size))
            self.stride = config.get('stride', self.stride)
            self.scales = config.get('scales', self.scales)
            self.hog_orientations = config.get('hog_orientations', self.hog_orientations)
            self.hog_pixels_per_cell = tuple(config.get('hog_pixels_per_cell', self.hog_pixels_per_cell))
            self.hog_cells_per_block = tuple(config.get('hog_cells_per_block', self.hog_cells_per_block))
            self.lbp_radius = config.get('lbp_radius', self.lbp_radius)
            self.lbp_n_points = config.get('lbp_n_points', self.lbp_n_points)
            self.gabor_frequencies = config.get('gabor_frequencies', self.gabor_frequencies)
            self.gabor_angles = config.get('gabor_angles', self.gabor_angles)
            self.rf_params = config.get('rf_params', self.rf_params)
            self.nms_threshold = config.get('nms_threshold', self.nms_threshold)
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def load_image(self, image_path):
        """Load image from file"""
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_path = image_path
            if not hasattr(self, 'original_filename') or self.original_filename is None:
                self.original_filename = os.path.basename(image_path)
            return True
        return False

    def add_annotation(self, x, y, width, height):
        """Add annotation to current image"""
        self.annotations.append({
            'x': x, 'y': y, 'width': width, 'height': height, 'class': 'ant'
        })

    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []

    def save_annotations_to_file(self):
        """Save annotations to JSON file"""
        if self.original_filename is None:
            fname = "annotations.json"
        else:
            base = os.path.splitext(os.path.basename(self.original_filename))[0]
            fname = base + "_annotations.json"
        data = {
            'image_path': self.original_filename,
            'annotations': self.annotations
        }
        json_data = json.dumps(data, indent=2)
        return fname, json_data

    def visualize_predictions(self, image, boxes, scores, min_score=None, color=(255, 0, 0), thickness=2):
        """
        Visualize predictions on image
        
        Args:
            image: Input image
            boxes: Detected bounding boxes
            scores: Confidence scores
            min_score: Minimum confidence score to display
            color: Box color as (R, G, B)
            thickness: Line thickness
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if min_score is not None and score < min_score:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Ant: {score:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - baseline), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return vis_image


# ---------------------------
# Global Session State Setup
# ---------------------------
if 'detector' not in st.session_state:
    st.session_state.detector = RandomForestAntDetector()
if 'canvas_annotations' not in st.session_state:
    st.session_state.canvas_annotations = []
if 'training_sets' not in st.session_state:
    st.session_state.training_sets = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ---------------------------
# Annotation Interface
# ---------------------------
def annotation_interface():
    if st.session_state.detector.image is None:
        st.warning("Please upload an image first.")
        return []
    
    if isinstance(st.session_state.detector.image, np.ndarray):
        pil_image = Image.fromarray(st.session_state.detector.image).convert("RGB")
    else:
        pil_image = st.session_state.detector.image.convert("RGB")
    
    img_width, img_height = pil_image.size
    
    st.markdown("""<style>.canvas-container { overflow-x: auto; }</style>""", unsafe_allow_html=True)
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFA500",
        background_image=pil_image,
        height=img_height,
        width=img_width,
        drawing_mode="rect",
        key="annotation_canvas",
        update_streamlit=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        st.session_state.canvas_annotations = []
        for obj in objects:
            if obj.get("type") == "rect":
                x = obj.get("left", 0)
                y = obj.get("top", 0)
                width = obj.get("width", 0) * obj.get("scaleX", 1)
                height = obj.get("height", 0) * obj.get("scaleY", 1)
                st.session_state.canvas_annotations.append({
                    "x": float(x), "y": float(y),
                    "width": float(width), "height": float(height)
                })
        
        st.session_state.detector.annotations = st.session_state.canvas_annotations
    
    if st.session_state.detector.annotations:
        fname, json_data = st.session_state.detector.save_annotations_to_file()
        st.download_button(
            label="Download Annotations", 
            data=json_data, 
            file_name=fname, 
            mime="application/json"
        )
    
    return st.session_state.canvas_annotations

# ---------------------------
# Load Training Data
# ---------------------------
def load_training_data(training_sets):
    """Load training data from the provided training sets"""
    X_list = []
    annotations_list = []
    
    for ts in training_sets:
        image_file = ts.get("image_file")
        annotation_file = ts.get("annotation_file")
        if image_file is None or annotation_file is None:
            continue
        
        try:
            # Load and process image
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            X_list.append(img)
            
            # Load annotations
            annotation_file.seek(0)
            data = json.loads(annotation_file.read().decode("utf-8"))
            
            img_annotations = []
            for ann in data.get("annotations", []):
                # Normalize coordinates to 0-1 range
                x1 = ann['x'] / w
                y1 = ann['y'] / h
                x2 = (ann['x'] + ann['width']) / w
                y2 = (ann['y'] + ann['height']) / h
                
                x1 = max(0.0, min(x1, 1.0))
                y1 = max(0.0, min(y1, 1.0))
                x2 = max(0.0, min(x2, 1.0))
                y2 = max(0.0, min(y2, 1.0))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                img_annotations.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': 1  # 1 = ant
                })
            
            annotations_list.append(img_annotations)
            
        except Exception as e:
            st.error(f"Error processing training set: {e}")
    
    if not X_list:
        return None, None
    
    return X_list, annotations_list

# ---------------------------
# Main Application
# ---------------------------
def main():
    st.title("🐜 Random Forest Ant Detection System")
    st.sidebar.header("Controls")
    
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Image Upload", "Annotation", "Train Model", "Prediction"],
        key="app_mode_selectbox"
    )
    
    # 1) IMAGE UPLOAD
    if app_mode == "Image Upload":
        st.header("Upload Image for Annotation")
        
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_filename = tmp_file.name
            
            if st.session_state.detector.load_image(temp_filename):
                st.session_state.detector.original_filename = uploaded_file.name
                st.success("Image loaded successfully!")
                st.image(st.session_state.detector.image, caption="Uploaded Image", use_column_width=True)
                st.session_state.canvas_annotations = []
            else:
                st.error("Failed to load image.")

    # 2) ANNOTATION
    elif app_mode == "Annotation":
        st.header("Annotate Image")
        _ = annotation_interface()
        st.info("Draw bounding boxes around all ants in the image. Then download the annotations file for later training use.")

    # 3) TRAIN MODEL
    elif app_mode == "Train Model":
        st.header("Train Random Forest Model")

        # Step A: Add training sets
        st.subheader("Training Sets (Each set: an image & its annotation file)")
        
        if st.button("Add Additional Training Set"):
            st.session_state.training_sets.append({"image_file": None, "annotation_file": None})

        for idx, ts in enumerate(st.session_state.training_sets):
            st.markdown(f"**Training Set {idx+1}**")
            col1, col2 = st.columns(2)
            
            with col1:
                image_file = st.file_uploader(
                    f"Image for set {idx+1}", type=['jpg', 'jpeg', 'png'],
                    key=f"ts_image_{idx}"
                )
            
            with col2:
                annotation_file = st.file_uploader(
                    f"Annotation (JSON) for set {idx+1}", type=["json"],
                    key=f"ts_ann_{idx}"
                )
            
            st.session_state.training_sets[idx]["image_file"] = image_file
            st.session_state.training_sets[idx]["annotation_file"] = annotation_file

        # Step B: Load training data
        st.write("---")
        if st.button("Load Training Data"):
            X, annotations = load_training_data(st.session_state.training_sets)
            if X is None or not X:
                st.error("No valid training data loaded.")
                st.session_state.data_loaded = False
            else:
                total_annotations = sum(len(anns) for anns in annotations)
                st.success(f"Training data loaded successfully! {len(X)} images with {total_annotations} total ant annotations.")
                st.session_state.data_loaded = True
                st.session_state.X = X
                st.session_state.annotations = annotations
                st.session_state.model_trained = False
                
                # Display sample from loaded data
                if len(X) > 0:
                    sample_idx = 0
                    sample_img = X[sample_idx]
                    sample_anns = annotations[sample_idx]
                    
                    fig, ax = plt.subplots(1, figsize=(10, 10))
                    ax.imshow(sample_img)
                    
                    h, w = sample_img.shape[:2]
                    
                    for ann in sample_anns:
                        x1, y1, x2, y2 = ann['bbox']
                        x1 = int(x1 * w)
                        y1 = int(y1 * h)
                        x2 = int(x2 * w)
                        y2 = int(y2 * h)
                        
                        rect = patches.Rectangle(
                            (x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)
                    
                    ax.set_title(f"Sample Image with {len(sample_anns)} Annotations")
                    st.pyplot(fig)

        # Step C: Model Configuration
        if st.session_state.data_loaded:
            st.subheader("Random Forest Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Detection Parameters")
                patch_size = st.selectbox(
                    "Patch Size", 
                    [(16, 16), (24, 24), (32, 32), (48, 48)], 
                    index=2,
                    format_func=lambda x: f"{x[0]}×{x[1]} pixels",
                    help="The size of the image patch that the model looks at to detect an ant. Smaller patches can detect smaller ants but may have more false positives. Larger patches are more accurate but may miss small ants."
                )
                st.session_state.detector.patch_size = patch_size
                
                stride = st.slider(
                    "Detection Stride", 
                    min_value=4, 
                    max_value=32, 
                    value=8,
                    help="How many pixels the detection window moves between checks. Smaller values are more thorough but slower (checks more positions). Larger values are faster but might miss some ants."
                )
                st.session_state.detector.stride = stride
                
                scales = st.multiselect(
                    "Detection Scales",
                    [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    default=[0.5, 1.0, 1.5],
                    help="The model will resize the image to these scales to detect ants of different sizes. More scales = better detection of various ant sizes but slower processing."
                )
                if scales:
                    st.session_state.detector.scales = sorted(scales)
            
            with col2:
                st.write("#### Feature Extraction")
                
                # HOG parameters
                hog_orientations = st.slider(
                    "HOG Orientations", 6, 12, 9,
                    help="Number of gradient directions to analyze. This helps the model recognize ant shapes from different angles. More orientations = better shape recognition but slower processing."
                )
                st.session_state.detector.hog_orientations = hog_orientations
                
                # LBP parameters
                lbp_radius = st.slider(
                    "LBP Radius", 1, 5, 3,
                    help="Size of the area around each pixel used for texture analysis. Larger radius captures more texture detail but may be slower."
                )
                lbp_n_points = st.slider(
                    "LBP Points", 8, 32, 24, step=8,
                    help="Number of points sampled around each pixel for texture analysis. More points = finer texture detection."
                )
                st.session_state.detector.lbp_radius = lbp_radius
                st.session_state.detector.lbp_n_points = lbp_n_points
                
                # NMS threshold
                nms_threshold = st.slider(
                    "Non-Maximum Suppression Threshold", 
                    0.1, 0.9, 0.3, 0.1,
                    help="Controls how much overlap is allowed between detected ants. Lower values (0.1-0.3) remove more overlapping detections, keeping only the best one. Higher values (0.7-0.9) keep more overlapping detections."
                )
                st.session_state.detector.nms_threshold = nms_threshold

        # Step D: Train Model
        if st.session_state.data_loaded:
            st.subheader("Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tune_hyperparameters = st.checkbox(
                    "Tune Hyperparameters", 
                    value=True,
                    help="Let the system automatically find the best settings for the Random Forest model. This takes longer but usually gives better results. Uncheck to manually set parameters."
                )
                
                if not tune_hyperparameters:
                    st.write("#### Manual Random Forest Parameters")
                    n_estimators = st.slider(
                        "Number of Trees", 10, 200, 100, 10,
                        help="How many decision trees to use. More trees = better accuracy but slower training and prediction."
                    )
                    max_depth = st.selectbox(
                        "Max Depth", [None, 10, 20, 30, 50], index=0,
                        help="Maximum depth of each tree. None = no limit (trees can grow until perfect). Smaller values prevent overfitting but may reduce accuracy."
                    )
                    min_samples_split = st.slider(
                        "Min Samples Split", 2, 20, 2,
                        help="Minimum number of samples required to split a tree node. Higher values prevent overfitting but may reduce accuracy."
                    )
                    min_samples_leaf = st.slider(
                        "Min Samples Leaf", 1, 10, 1,
                        help="Minimum number of samples required at a leaf node. Higher values create simpler trees that generalize better."
                    )
                    
                    st.session_state.detector.rf_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'random_state': 42
                    }
            
            with col2:
                st.write("#### Training Info")
                st.info("""
                **Why Random Forest works well for ant detection:**
                
                • **Works with small datasets**: Unlike deep learning, it can train effectively with just 10-20 annotated images
                • **No overfitting**: Automatically prevents memorizing training data
                • **Fast training**: Takes minutes, not hours
                • **Interpretable**: Can show which image features are most important
                • **CPU-friendly**: No expensive GPU required
                • **Robust**: Handles varying image quality well
                """)
            
            # Train Model Button
            if st.button("Train Random Forest Model"):
                with st.spinner("Training Random Forest model..."):
                    try:
                        results = st.session_state.detector.train_model(
                            st.session_state.X,
                            st.session_state.annotations,
                            tune_hyperparameters=tune_hyperparameters
                        )
                        
                        st.success("Training complete!")
                        st.session_state.model_trained = True
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Training Accuracy", f"{results['accuracy']:.3f}")
                            st.metric("Training Samples", results['n_samples'])
                            st.metric("Feature Vector Size", results['n_features'])
                        
                        with col2:
                            # Feature importance plot
                            if 'feature_importance' in results:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Get top 20 most important features
                                importance = results['feature_importance']
                                top_indices = np.argsort(importance)[-20:]
                                top_importance = importance[top_indices]
                                
                                ax.barh(range(len(top_importance)), top_importance)
                                ax.set_ylabel("Feature Index")
                                ax.set_xlabel("Importance")
                                ax.set_title("Top 20 Most Important Features")
                                
                                st.pyplot(fig)
                        
                        # Model summary
                        st.subheader("Model Summary")
                        if hasattr(st.session_state.detector.model, 'get_params'):
                            params = st.session_state.detector.model.get_params()
                            st.write("**Final Model Parameters:**")
                            for key, value in params.items():
                                st.write(f"- {key}: {value}")
                    
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        import traceback
                        st.error(traceback.format_exc())
            
            # Save model option
            if st.session_state.model_trained:
                st.subheader("Save Trained Model")
                
                model_name = st.text_input("Model Name", "random_forest_ant_detector.pkl")
                
                # Directly prepare the model for download without a button click
                model_buffer = st.session_state.detector.save_model_to_bytes()
                if model_buffer:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=model_buffer.getvalue(),
                        file_name=model_name,
                        mime="application/octet-stream"
                    )
                    st.success("Model ready for download! Click the button above to save.")
                else:
                    st.error("Failed to prepare model for download.")

    # 4) PREDICTION
    elif app_mode == "Prediction":
        st.header("Random Forest Ant Detection")
        
        # Model loading
        st.subheader("Load Model")
        
        uploaded_model = st.file_uploader(
            "Upload a trained model file (.pkl)", 
            type=['pkl'],
            help="Upload a model file that was previously trained and downloaded from this app."
        )
        
        if uploaded_model is not None:
            if st.button("Load Uploaded Model"):
                with st.spinner("Loading model..."):
                    model_bytes = uploaded_model.read()
                    if st.session_state.detector.load_model_from_bytes(model_bytes):
                        st.success(f"Model loaded successfully!")
                        st.write(f"**Configuration:** {st.session_state.detector.patch_size} patch size, "
                               f"stride {st.session_state.detector.stride}, "
                               f"scales {st.session_state.detector.scales}")
                    else:
                        st.error("Failed to load model.")
        
        # Image upload for prediction
        st.subheader("Upload Image for Detection")
        
        pred_image = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'png'], key="pred_image")
        
        # Detection parameters
        st.write("#### Detection Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Detection Confidence Threshold", 
                0.0, 1.0, 0.6, 0.05,
                help="How confident the model needs to be to mark something as an ant. Lower values (0.3-0.5) find more ants but may have false positives. Higher values (0.7-0.9) are more accurate but may miss some ants."
            )
            
            nms_threshold = st.slider(
                "Overlap Removal Threshold", 
                0.0, 1.0, st.session_state.detector.nms_threshold, 0.05,
                help="Controls how overlapping detections are handled. Lower values (0.1-0.3) aggressively remove overlaps, keeping only the best detection. Higher values (0.5-0.9) allow more overlapping detections."
            )
        
        with col2:
            show_all_detections = st.checkbox(
                "Show All Detections (Before Overlap Removal)",
                value=False,
                help="Show all potential ant detections before removing overlaps. Useful for debugging if ants are being missed."
            )
            
            fast_mode = st.checkbox(
                "Fast Detection Mode",
                value=True,
                help="Speed up detection by checking fewer positions and scales. Good for quick results but may miss some ants."
            )
            st.session_state.detector.fast_mode = fast_mode
            
            detection_color = st.color_picker("Detection Box Color", "#FF0000")
            # Convert hex to RGB
            detection_color_rgb = tuple(int(detection_color[i:i+2], 16) for i in (1, 3, 5))
        
        # Apply detection
        if pred_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(pred_image.getvalue())
                temp_filename = tmp_file.name
            
            if st.session_state.detector.load_image(temp_filename):
                st.session_state.detector.original_filename = pred_image.name
                st.image(st.session_state.detector.image, caption="Image for Detection", use_column_width=True)
                
                if st.session_state.detector.model is not None:
                    if st.button("Detect Ants"):
                        with st.spinner("Running Random Forest detection..."):
                            try:
                                image = st.session_state.detector.image
                                
                                boxes, scores, all_detections = st.session_state.detector.predict(
                                    image, 
                                    confidence_threshold=confidence_threshold,
                                    nms_threshold=nms_threshold
                                )
                                
                                if not boxes:
                                    st.info(f"No ants detected above confidence threshold {confidence_threshold:.2f}")
                                    
                                    if all_detections:
                                        all_scores = [det['score'] for det in all_detections]
                                        st.subheader("Detection Score Analysis")
                                        
                                        fig, ax = plt.subplots(figsize=(10, 4))
                                        ax.hist(all_scores, bins=20, range=(0, 1))
                                        ax.axvline(x=confidence_threshold, color='r', linestyle='--', 
                                                 label=f'Threshold ({confidence_threshold})')
                                        ax.set_xlabel("Confidence Score")
                                        ax.set_ylabel("Number of Detections")
                                        ax.set_title("All Detection Scores")
                                        ax.legend()
                                        st.pyplot(fig)
                                        
                                        top_scores = sorted(all_scores, reverse=True)[:10]
                                        st.write("Top 10 scores:", [f"{s:.3f}" for s in top_scores])
                                        st.info("💡 Try lowering the confidence threshold to include more detections.")
                                else:
                                    # Visualize detections
                                    vis_image = st.session_state.detector.visualize_predictions(
                                        image, boxes, scores, 
                                        min_score=confidence_threshold,
                                        color=detection_color_rgb
                                    )
                                    
                                    st.image(vis_image, caption="Detection Results", use_column_width=True)
                                    
                                    # Results table
                                    st.subheader(f"Detection Results: {len(boxes)} Ants Found")
                                    results_df = {
                                        "Detection #": list(range(1, len(boxes) + 1)),
                                        "Confidence": [f"{s:.3f}" for s in scores],
                                        "Position (x1,y1,x2,y2)": [f"({int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])})" for b in boxes],
                                        "Width × Height": [f"{int(b[2] - b[0])} × {int(b[3] - b[1])}" for b in boxes]
                                    }
                                    st.dataframe(results_df)
                                    
                                    # Show all detections if requested
                                    if show_all_detections and all_detections:
                                        st.subheader(f"All Detections Before Overlap Removal: {len(all_detections)}")
                                        
                                        # Visualize all detections
                                        all_boxes = [det['bbox'] for det in all_detections]
                                        all_scores = [det['score'] for det in all_detections]
                                        
                                        vis_image_all = st.session_state.detector.visualize_predictions(
                                            image, all_boxes, all_scores,
                                            min_score=confidence_threshold,
                                            color=(0, 255, 0),  # Green for all detections
                                            thickness=1
                                        )
                                        
                                        st.image(vis_image_all, caption="All Detections (Before Overlap Removal)", use_column_width=True)
                                    
                                    # Detection statistics
                                    st.subheader("Detection Statistics")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if boxes:
                                            widths = [b[2] - b[0] for b in boxes]
                                            heights = [b[3] - b[1] for b in boxes]
                                            
                                            st.write(f"**Size Statistics:**")
                                            st.write(f"- Avg width: {np.mean(widths):.1f}px")
                                            st.write(f"- Avg height: {np.mean(heights):.1f}px")
                                            st.write(f"- Width range: {min(widths):.0f}-{max(widths):.0f}px")
                                            st.write(f"- Height range: {min(heights):.0f}-{max(heights):.0f}px")
                                    
                                    with col2:
                                        st.write(f"**Detection Statistics:**")
                                        st.write(f"- Detections found: {len(boxes)}")
                                        st.write(f"- Avg confidence: {np.mean(scores):.3f}")
                                        st.write(f"- Min confidence: {min(scores):.3f}")
                                        st.write(f"- Max confidence: {max(scores):.3f}")
                                        if all_detections:
                                            st.write(f"- Total before overlap removal: {len(all_detections)}")
                                    
                                    # Score distribution
                                    if len(scores) > 1:
                                        st.subheader("Score Distribution")
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        ax.hist(scores, bins=min(10, len(scores)), alpha=0.7)
                                        ax.axvline(x=confidence_threshold, color='r', linestyle='--', 
                                                 label=f'Threshold ({confidence_threshold})')
                                        ax.set_xlabel("Confidence Score")
                                        ax.set_ylabel("Number of Detections")
                                        ax.set_title("Final Detection Scores")
                                        ax.legend()
                                        st.pyplot(fig)
                            
                            except Exception as e:
                                st.error(f"Error during detection: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                else:
                    st.warning("Please load a model first.")
        
        # Help information
        with st.expander("🔍 Detection Tips for Best Results"):
            st.write("""
            #### Getting the Best Detection Results
            
            **1. Confidence Threshold**
            - Start with 0.5-0.7 for balanced results
            - Lower to 0.3-0.4 if missing ants
            - Raise to 0.8-0.9 if too many false positives
            
            **2. Image Quality Tips**
            - Use well-lit images with good contrast
            - Ants should be clearly visible against the background
            - Avoid blurry or very dark images
            
            **3. If Ants Are Being Missed:**
            - Turn off "Fast Detection Mode"
            - Lower the confidence threshold
            - Check if your training data included similar-sized ants
            
            **4. If Getting Too Many False Positives:**
            - Increase the confidence threshold
            - Lower the overlap removal threshold
            - Ensure training data has diverse background examples
            
            **5. Model Performance:**
            - Random Forest works best with consistent ant sizes
            - Train with 10-20 diverse images for best results
            - Include various backgrounds in training data
            """)

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    main()
