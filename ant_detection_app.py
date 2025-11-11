#cd ~/Desktop/AntDetector
#python3 ant_detection_app.py

#!/usr/bin/env python3
"""
Random Forest Ant Detection System - Enhanced Desktop Application
Complete port from Streamlit with all original functionality
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import json
import pickle
import io
import os
import threading
import queue
from datetime import datetime
import pandas as pd
from pathlib import Path
import gc
import psutil
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, precision_recall_curve, average_precision_score, 
                           confusion_matrix, cohen_kappa_score)
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor

class RandomForestAntDetector:
    """Complete Random Forest Ant Detector - matches original Streamlit version"""
    
    def __init__(self):
        self.model = None
        self.labels = ['background', 'ant']
        self.image = None
        self.image_path = None
        self.original_filename = None
        self.annotations = []
        self.current_model_path = None
        self.training_images = []
        self.training_annotations = []
        
        # Memory optimization settings (from original)
        available_memory_gb = psutil.virtual_memory().total / (1024**3)
        if available_memory_gb <= 8:
            self.max_patches_per_image = 1000
            self.batch_size = 50
            max_estimators = 25
        elif available_memory_gb <= 16:
            self.max_patches_per_image = 2000
            self.batch_size = 100
            max_estimators = 50
        else:
            self.max_patches_per_image = 3000
            self.batch_size = 200
            max_estimators = 100
        
        # Feature extraction configuration (from original)
        self.patch_size = (32, 32)
        self.stride = 16
        self.scales = [0.75, 1.0, 1.25]
        self.fast_mode = True
        
        # HOG parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_n_points = 24
        
        # Gabor filter parameters
        self.gabor_frequencies = [0.3]
        self.gabor_angles = [0, np.pi/2]
        
        # Random Forest parameters (from original)
        self.rf_params = {
            'n_estimators': min(100, max_estimators),
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': 1,
            'max_features': 'sqrt'
        }
        
        # Center-point suppression parameters
        self.distance_threshold = 0.05
        self.use_normalized_distance = True
        self.confidence_threshold = 0.6
        
    def extract_features(self, patch):
        """Extract features with descriptive names - matches original exactly"""
        if len(patch.shape) == 3:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray_patch = patch
        
        features = []
        feature_names = []
        
        # 1. HOG features
        try:
            hog_features = hog(
                gray_patch,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm='L2-Hys',
                feature_vector=True,
                transform_sqrt=True
            )
            features.extend(hog_features.astype(np.float32))
            
            for i in range(len(hog_features)):
                feature_names.append(f"HOG_gradient_orientation_{i % self.hog_orientations}_block_{i // self.hog_orientations}")
                
        except:
            default_hog_size = 324
            features.extend([0] * default_hog_size)
            for i in range(default_hog_size):
                feature_names.append(f"HOG_gradient_orientation_{i % self.hog_orientations}_block_{i // self.hog_orientations}")
        
        # 2. Local Binary Pattern (LBP) features
        try:
            lbp = local_binary_pattern(gray_patch, self.lbp_n_points, self.lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist.astype(np.float32)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist)
            
            for i in range(len(lbp_hist)):
                feature_names.append(f"LBP_texture_pattern_bin_{i}")
                
        except:
            features.extend([0] * 10)
            for i in range(10):
                feature_names.append(f"LBP_texture_pattern_bin_{i}")
        
        # 3. Gabor filter responses
        for freq_idx, freq in enumerate(self.gabor_frequencies):
            for angle_idx, angle in enumerate(self.gabor_angles):
                try:
                    real, _ = gabor(gray_patch, frequency=freq, theta=angle)
                    gabor_mean = np.mean(real).astype(np.float32)
                    gabor_std = np.std(real).astype(np.float32)
                    gabor_max = np.max(real).astype(np.float32)
                    
                    features.extend([gabor_mean, gabor_std, gabor_max])
                    
                    angle_deg = int(np.degrees(angle))
                    feature_names.extend([
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_mean_response",
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_std_response", 
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_max_response"
                    ])
                    
                except:
                    features.extend([0, 0, 0])
                    angle_deg = int(np.degrees(angle)) if 'angle' in locals() else freq_idx * 45
                    feature_names.extend([
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_mean_response",
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_std_response",
                        f"Gabor_freq_{freq:.1f}_angle_{angle_deg}deg_max_response"
                    ])
        
        # 4. Basic statistical features
        mean_intensity = np.mean(gray_patch).astype(np.float32)
        std_intensity = np.std(gray_patch).astype(np.float32)
        max_intensity = np.max(gray_patch).astype(np.float32)
        min_intensity = np.min(gray_patch).astype(np.float32)
        median_intensity = np.median(gray_patch).astype(np.float32)
        
        features.extend([mean_intensity, std_intensity, max_intensity, min_intensity, median_intensity])
        feature_names.extend([
            "pixel_intensity_mean",
            "pixel_intensity_std_deviation", 
            "pixel_intensity_maximum",
            "pixel_intensity_minimum",
            "pixel_intensity_median"
        ])
        
        # 5. Edge features
        try:
            edges = cv2.Canny(gray_patch, 50, 150)
            edge_density = (np.sum(edges > 0) / edges.size).astype(np.float32)
            features.append(edge_density)
            feature_names.append("edge_density_canny")
        except:
            features.append(0)
            feature_names.append("edge_density_canny")
        
        # 6. Texture features
        try:
            dx = np.diff(gray_patch, axis=1)
            dy = np.diff(gray_patch, axis=0)
            texture_energy = (np.mean(dx**2) + np.mean(dy**2)).astype(np.float32)
            features.append(texture_energy)
            feature_names.append("texture_gradient_energy")
        except:
            features.append(0)
            feature_names.append("texture_gradient_energy")
        
        # Store feature names
        if not hasattr(self, 'feature_names'):
            self.feature_names = feature_names
        
        return np.array(features, dtype=np.float32)
    
    def extract_patches_and_labels(self, images, annotations_list, progress_callback=None):
        """Extract training patches with memory management and proper class balance"""
        X_features = []
        y_labels = []
        
        for img_idx, (image, annotations) in enumerate(zip(images, annotations_list)):
            if progress_callback:
                progress_callback(img_idx, len(images), f"Processing image {img_idx + 1}/{len(images)}")
            
            h, w = image.shape[:2]
            
            # Convert annotations to pixel coordinates
            ant_boxes = []
            for ann in annotations:
                if isinstance(ann, dict) and 'bbox' in ann:
                    # Normalized coordinates
                    x1, y1, x2, y2 = ann['bbox']
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                else:
                    # Pixel coordinates
                    x = ann['x']
                    y = ann['y']
                    width = ann['width']
                    height = ann['height']
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + width), int(y + height)
                
                ant_boxes.append([x1, y1, x2, y2])
            
            # Cap positive samples at 4000 per image
            max_positive_per_image = 4000
            if len(ant_boxes) > max_positive_per_image:
                # Randomly sample 4000 ants
                import random
                ant_boxes = random.sample(ant_boxes, max_positive_per_image)
            
            # Extract positive samples (ants)
            positive_count = 0
            
            for box in ant_boxes:
                x1, y1, x2, y2 = box
                ant_center_x = (x1 + x2) // 2
                ant_center_y = (y1 + y2) // 2
                
                # Extract patch around ant center
                patch_x = max(0, min(w - self.patch_size[0], ant_center_x - self.patch_size[0]//2))
                patch_y = max(0, min(h - self.patch_size[1], ant_center_y - self.patch_size[1]//2))
                
                patch = image[patch_y:patch_y + self.patch_size[1], patch_x:patch_x + self.patch_size[0]]
                
                if patch.shape[:2] == self.patch_size:
                    features = self.extract_features(patch)
                    if len(features) > 0:
                        X_features.append(features)
                        y_labels.append(1)  # Ant
                        positive_count += 1
            
            # Extract negative samples (background)
            # Use 1:1 ratio with positive samples for better balance
            negative_target = positive_count
            negative_count = 0
            attempts = 0
            max_attempts = negative_target * 20  # Increased attempts to find good negatives
            
            while negative_count < negative_target and attempts < max_attempts:
                attempts += 1
                
                rand_x = np.random.randint(0, max(1, w - self.patch_size[0]))
                rand_y = np.random.randint(0, max(1, h - self.patch_size[1]))
                
                # Check if patch overlaps with any ant (with margin)
                patch_center_x = rand_x + self.patch_size[0] // 2
                patch_center_y = rand_y + self.patch_size[1] // 2
                
                is_ant = False
                # Increased margin to ensure clearer background samples
                margin = self.patch_size[0]  # Full patch size as margin
                for box in ant_boxes:
                    bx1, by1, bx2, by2 = box
                    if (bx1 - margin) <= patch_center_x <= (bx2 + margin) and (by1 - margin) <= patch_center_y <= (by2 + margin):
                        is_ant = True
                        break
                
                if not is_ant:
                    patch = image[rand_y:rand_y + self.patch_size[1], rand_x:rand_x + self.patch_size[0]]
                    
                    if patch.shape[:2] == self.patch_size:
                        features = self.extract_features(patch)
                        if len(features) > 0:
                            X_features.append(features)
                            y_labels.append(0)  # Background
                            negative_count += 1
            
            # Log class balance for this image
            if progress_callback:
                progress_callback(img_idx, len(images), 
                                f"Image {img_idx + 1}: {positive_count} ants, {negative_count} background")
            
            # Periodic garbage collection
            if img_idx % 5 == 0:
                gc.collect()
        
        X = np.array(X_features, dtype=np.float32)
        y = np.array(y_labels)
        
        # Final balance check
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        print(f"\nFinal dataset balance: {pos_count} positive, {neg_count} negative samples")
        print(f"Ratio: {pos_count/(pos_count + neg_count):.2%} positive")
        
        return X, y
    
    def train_model(self, images, annotations, tune_hyperparameters=True, progress_callback=None):
        """Train model with hyperparameter tuning - matches original"""
        try:
            # Validate inputs
            if not images or not annotations:
                raise ValueError("No images or annotations provided!")
            
            if len(images) != len(annotations):
                raise ValueError(f"Mismatch: {len(images)} images but {len(annotations)} annotation sets!")
            
            total_ants = sum(len(ann_set) for ann_set in annotations)
            if total_ants == 0:
                raise ValueError("No ant annotations found in any image!")
            
            # Extract features
            X, y = self.extract_patches_and_labels(images, annotations, progress_callback)
            
            if len(X) == 0:
                raise ValueError("No training samples extracted!")
            
            # Check class balance - updated for 1:1 ratio
            pos_ratio = np.sum(y) / len(y)
            if pos_ratio < 0.35 or pos_ratio > 0.65:
                raise ValueError(f"Class imbalance detected: {pos_ratio:.1%} positive samples. "
                               f"Aim for 40-60% for best results.")
            
            # Train model
            if tune_hyperparameters:
                param_distributions = {
                    'n_estimators': [25, 50, min(75, self.rf_params['n_estimators'])],
                    'max_depth': [6, 8, 10],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 4, 8],
                    'max_features': ['sqrt', 'log2']
                }
                
                rf = RandomForestClassifier(random_state=42, n_jobs=1, bootstrap=True)
                cv_folds = min(3, max(2, len(X) // 100))
                
                search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=param_distributions,
                    n_iter=6,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=1,
                    random_state=42,
                    return_train_score=False
                )
                
                search.fit(X, y)
                self.model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
            else:
                self.model = RandomForestClassifier(**self.rf_params)
                self.model.fit(X, y)
                best_params = self.rf_params
                best_score = None
            
            # Calculate metrics
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            # Store actual sample counts before cleanup
            actual_positive_samples = int(np.sum(y == 1))
            actual_negative_samples = int(np.sum(y == 0))
            
            # Clean up training data
            del X, y
            gc.collect()
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'best_params': best_params,
                'best_score': best_score,
                'feature_importance': self.model.feature_importances_,
                'n_samples': len(images),
                'n_features': len(self.model.feature_importances_),
                'positive_samples': actual_positive_samples,
                'negative_samples': actual_negative_samples
            }
            
        except Exception as e:
            self.model = None
            raise e
    
    def sliding_window_detection(self, image, confidence_threshold=None, progress_callback=None):
        """Sliding window detection with batch processing - matches original"""
        if self.model is None:
            return []
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        detections = []
        h, w = image.shape[:2]
        
        # Use fast mode settings if enabled
        scales = [1.0] if self.fast_mode else self.scales
        stride = max(16, self.stride) if self.fast_mode else self.stride
        
        # Calculate total operations
        total_operations = 0
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h >= self.patch_size[1] and new_w >= self.patch_size[0]:
                y_steps = ((new_h - self.patch_size[1]) // stride) + 1
                x_steps = ((new_w - self.patch_size[0]) // stride) + 1
                total_operations += y_steps * x_steps
        
        if total_operations == 0:
            return []
        
        operations_completed = 0
        
        # Multi-scale detection
        for scale_idx, scale in enumerate(scales):
            if progress_callback:
                progress_callback(scale_idx, len(scales), f"Processing scale {scale}")
            
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h < self.patch_size[1] or new_w < self.patch_size[0]:
                continue
                
            resized_image = cv2.resize(image, (new_w, new_h))
            
            # Batch processing
            batch_patches = []
            batch_positions = []
            batch_size = min(100, self.batch_size)
            
            for y in range(0, new_h - self.patch_size[1] + 1, stride):
                for x in range(0, new_w - self.patch_size[0] + 1, stride):
                    patch = resized_image[y:y + self.patch_size[1], x:x + self.patch_size[0]]
                    
                    if patch.shape[:2] != self.patch_size:
                        operations_completed += 1
                        continue
                    
                    batch_patches.append(patch)
                    batch_positions.append((x, y, scale))
                    
                    if len(batch_patches) >= batch_size:
                        detections.extend(self._process_patch_batch(
                            batch_patches, batch_positions, confidence_threshold
                        ))
                        batch_patches = []
                        batch_positions = []
                        gc.collect()
                    
                    operations_completed += 1
            
            # Process remaining patches
            if batch_patches:
                detections.extend(self._process_patch_batch(
                    batch_patches, batch_positions, confidence_threshold
                ))
            
            del resized_image
            gc.collect()
        
        return detections
    
    def _process_patch_batch(self, patches, positions, confidence_threshold):
        """Process batch of patches efficiently"""
        batch_detections = []
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
        
        try:
            feature_matrix = np.array(feature_vectors, dtype=np.float32)
            probabilities = self.model.predict_proba(feature_matrix)
            
            for prob_idx, prob in enumerate(probabilities):
                original_idx = valid_indices[prob_idx]
                x, y, scale = positions[original_idx]
                
                ant_prob = prob[1]
                
                if ant_prob >= confidence_threshold:
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + self.patch_size[0]) / scale)
                    y2 = int((y + self.patch_size[1]) / scale)
                    
                    batch_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': ant_prob,
                        'scale': scale
                    })
            
            del feature_matrix, probabilities
            
        except Exception:
            # Fallback to individual processing
            for i, patch in enumerate(patches):
                if i not in valid_indices:
                    continue
                    
                try:
                    features = self.extract_features(patch)
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
    
    def center_point_suppression(self, detections, distance_threshold=None, image_width=None, image_height=None, use_normalized=True):
        """Center point based non-maximum suppression - matches original"""
        if not detections:
            return []
        
        if use_normalized and image_width and image_height:
            image_diagonal = np.sqrt(image_width**2 + image_height**2)
            threshold_pixels = distance_threshold * image_diagonal
        else:
            threshold_pixels = distance_threshold if distance_threshold else 50
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['score'] for det in detections])
        
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ])
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            current_center = centers[current]
            remaining_centers = centers[indices[1:]]
            distances = np.sqrt(np.sum((remaining_centers - current_center)**2, axis=1))
            far_enough = distances >= threshold_pixels
            indices = indices[1:][far_enough]
        
        return [detections[i] for i in keep]
    
    def predict(self, image, confidence_threshold=None, distance_threshold=None, use_normalized_distance=True, progress_callback=None):
        """Main prediction function - matches original"""
        if self.model is None:
            return [], [], []
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if distance_threshold is None:
            distance_threshold = 0.05 if use_normalized_distance else 50
        
        self.image = image
        h, w = image.shape[:2]
        
        # Perform sliding window detection
        detections = self.sliding_window_detection(image, confidence_threshold, progress_callback)
        
        if not detections:
            return [], [], []
        
        # Apply center-point suppression
        filtered_detections = self.center_point_suppression(
            detections, 
            distance_threshold=distance_threshold,
            image_width=w,
            image_height=h,
            use_normalized=use_normalized_distance
        )
        
        boxes = [det['bbox'] for det in filtered_detections]
        scores = [det['score'] for det in filtered_detections]
        
        return boxes, scores, detections
    
    def save_model_to_file(self, filepath):
        """Save model to file"""
        if self.model is not None:
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
                    'distance_threshold': self.distance_threshold,
                    'use_normalized_distance': self.use_normalized_distance,
                    'confidence_threshold': self.confidence_threshold,
                    'max_patches_per_image': self.max_patches_per_image,
                    'batch_size': self.batch_size,
                    'feature_names': getattr(self, 'feature_names', None)
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)
            return True
        return False
    
    def load_model_from_file(self, filepath):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
            
            self.model = save_dict['model']
            config = save_dict.get('config', {})
            
            # Load all configuration
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
            self.distance_threshold = config.get('distance_threshold', self.distance_threshold)
            self.use_normalized_distance = config.get('use_normalized_distance', self.use_normalized_distance)
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            self.max_patches_per_image = config.get('max_patches_per_image', self.max_patches_per_image)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.feature_names = config.get('feature_names', None)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Check if there is an intersection
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def calculate_center_distance(self, pred_box, gt_box):
        """Calculate Euclidean distance between centers of two bounding boxes"""
        pred_center_x = (pred_box[0] + pred_box[2]) / 2
        pred_center_y = (pred_box[1] + pred_box[3]) / 2
        gt_center_x = (gt_box[0] + gt_box[2]) / 2
        gt_center_y = (gt_box[1] + gt_box[3]) / 2
        
        distance = np.sqrt((pred_center_x - gt_center_x)**2 + (pred_center_y - gt_center_y)**2)
        return distance

    def calculate_normalized_center_distance(self, pred_box, gt_box, image_width, image_height):
        """Calculate normalized center distance as percentage of image diagonal"""
        distance = self.calculate_center_distance(pred_box, gt_box)
        image_diagonal = np.sqrt(image_width**2 + image_height**2)
        normalized_distance = distance / image_diagonal
        return normalized_distance

    def evaluate_object_detection(self, pred_boxes, pred_scores, gt_boxes, distance_threshold=50, 
                                image_width=None, image_height=None, normalized_threshold=0.05):
        """
        Evaluate object detection performance using center point distance matching
        """
        if not pred_boxes:
            pred_boxes = []
            pred_scores = []
        if not gt_boxes:
            gt_boxes = []
        
        num_predictions = len(pred_boxes)
        num_ground_truth = len(gt_boxes)
        
        # Determine whether to use normalized or pixel thresholds
        use_normalized = (image_width is not None and image_height is not None)
        
        # Initialize arrays
        true_positives = np.zeros(num_predictions)
        false_positives = np.zeros(num_predictions)
        gt_matched = np.zeros(num_ground_truth, dtype=bool)
        match_distances = []
        
        # Sort predictions by confidence score (descending)
        if num_predictions > 0:
            sorted_indices = np.argsort(pred_scores)[::-1]
            sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
            sorted_pred_scores = [pred_scores[i] for i in sorted_indices]
        else:
            sorted_pred_boxes = []
            sorted_pred_scores = []
        
        # Match predictions to ground truth using center distance
        for pred_idx, (pred_box, pred_score) in enumerate(zip(sorted_pred_boxes, sorted_pred_scores)):
            best_distance = float('inf')
            best_gt_idx = -1
            
            # Find closest ground truth center
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue  # Already matched
                
                if use_normalized:
                    distance = self.calculate_normalized_center_distance(
                        pred_box, gt_box, image_width, image_height
                    )
                    threshold = normalized_threshold
                else:
                    distance = self.calculate_center_distance(pred_box, gt_box)
                    threshold = distance_threshold
                
                if distance < best_distance:
                    best_distance = distance
                    best_gt_idx = gt_idx
            
            # Check if best match is within threshold
            threshold = normalized_threshold if use_normalized else distance_threshold
            if best_distance <= threshold and best_gt_idx >= 0:
                true_positives[pred_idx] = 1
                gt_matched[best_gt_idx] = True
                match_distances.append(best_distance)
            else:
                false_positives[pred_idx] = 1
        
        # Calculate cumulative TP and FP for precision-recall curve
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)
        
        # Calculate precision and recall curves
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        recall = cum_tp / (num_ground_truth + 1e-10)
        
        # Calculate Average Precision (AP)
        ap = 0.0
        if num_ground_truth > 0:
            for t in np.arange(0, 1.1, 0.1):
                p_interp = np.max(precision[recall >= t]) if np.any(recall >= t) else 0
                ap += p_interp / 11
        
        # Final metrics
        total_tp = np.sum(true_positives)
        total_fp = np.sum(false_positives)
        total_fn = num_ground_truth - total_tp
        
        final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        final_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
        
        # Center-point specific metrics
        mean_match_distance = np.mean(match_distances) if match_distances else 0
        median_match_distance = np.median(match_distances) if match_distances else 0
        max_match_distance = np.max(match_distances) if match_distances else 0
        
        # Classification metrics for center point evaluation
        y_true = []
        y_pred = []
        
        # For each prediction, mark as correct/incorrect based on center distance
        for pred_idx in range(num_predictions):
            if true_positives[pred_idx] == 1:
                y_pred.append(1)  # Correct detection
                y_true.append(1)  # There was actually an ant here
            else:
                y_pred.append(1)  # Model predicted ant
                y_true.append(0)  # But center wasn't close enough to any GT
        
        # For each unmatched ground truth, add as missed detection
        for gt_idx in range(num_ground_truth):
            if not gt_matched[gt_idx]:
                y_pred.append(0)  # Model didn't detect
                y_true.append(1)  # But there was an ant
        
        # Calculate Kappa and Accuracy
        if len(y_true) > 0 and len(set(y_true)) > 1:
            try:
                from sklearn.metrics import cohen_kappa_score, accuracy_score
                kappa_score = cohen_kappa_score(y_true, y_pred)
                detection_accuracy = accuracy_score(y_true, y_pred)
            except:
                kappa_score = 0.0
                detection_accuracy = 0.0
        else:
            kappa_score = 0.0
            detection_accuracy = sum(np.array(y_true) == np.array(y_pred)) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            # Standard detection metrics
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': f1_score,
            'average_precision': ap,
            
            # Counts
            'true_positives': int(total_tp),
            'false_positives': int(total_fp),
            'false_negatives': int(total_fn),
            'num_predictions': num_predictions,
            'num_ground_truth': num_ground_truth,
            
            # Curves for plotting
            'precision_curve': precision,
            'recall_curve': recall,
            'pred_scores': sorted_pred_scores if sorted_pred_scores else [],
            
            # Classification metrics
            'kappa_score': kappa_score,
            'detection_accuracy': detection_accuracy,
            
            # Center-point specific metrics
            'mean_match_distance': mean_match_distance,
            'median_match_distance': median_match_distance,
            'max_match_distance': max_match_distance,
            'num_matches': len(match_distances),
            'match_distances': match_distances,
            
            # Evaluation config
            'distance_threshold': normalized_threshold if use_normalized else distance_threshold,
            'threshold_type': 'normalized' if use_normalized else 'pixels',
            'image_dimensions': f"{image_width}x{image_height}" if use_normalized else "N/A"
        }

    def visualize_test_results(self, image, pred_boxes, pred_scores, gt_boxes, 
                              distance_threshold=50, use_normalized=False, 
                              image_width=None, image_height=None):
        """
        Visualize test results showing TP, FP, FN with ground truth boxes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Calculate threshold for matching
        if use_normalized:
            image_diagonal = np.sqrt(image_width**2 + image_height**2)
            threshold_pixels = distance_threshold * image_diagonal
        else:
            threshold_pixels = distance_threshold
        
        # Find matches between predictions and ground truth
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)
        
        if pred_boxes:
            # Sort by confidence
            sorted_indices = np.argsort(pred_scores)[::-1]
            
            for orig_idx in sorted_indices:
                pred_box = pred_boxes[orig_idx]
                
                # Find best match
                best_distance = float('inf')
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                        
                    if use_normalized:
                        distance = self.calculate_normalized_center_distance(
                            pred_box, gt_box, image_width, image_height
                        )
                        threshold = distance_threshold
                    else:
                        distance = self.calculate_center_distance(pred_box, gt_box)
                        threshold = distance_threshold
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_gt_idx = gt_idx
                
                # Determine if match is valid
                is_match = (best_distance <= threshold and best_gt_idx >= 0)
                
                if is_match:
                    # True Positive - mark both as matched
                    gt_matched[best_gt_idx] = True
                    pred_matched[orig_idx] = True
        
        # Create visualization using matplotlib
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw ALL ground truth boxes as filled blue rectangles
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = gt_box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1, edgecolor='cyan', facecolor='cyan', alpha=0.3)
            ax.add_patch(rect)
        
        # Count results
        tp_count = sum(pred_matched)
        fp_count = len(pred_boxes) - tp_count
        fn_count = len(gt_boxes) - sum(gt_matched)
        
        # Draw True Positives (green)
        for i, (pred_box, is_matched) in enumerate(zip(pred_boxes, pred_matched)):
            if is_matched:
                x1, y1, x2, y2 = pred_box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='green', facecolor='none', alpha=0.8)
                ax.add_patch(rect)
        
        # Draw False Positives (orange)
        for i, (pred_box, is_matched) in enumerate(zip(pred_boxes, pred_matched)):
            if not is_matched:
                x1, y1, x2, y2 = pred_box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='orange', facecolor='none', alpha=0.8)
                ax.add_patch(rect)
        
        # Draw False Negatives (red outline on top of blue ground truth)
        for i, (gt_box, is_matched) in enumerate(zip(gt_boxes, gt_matched)):
            if not is_matched:
                x1, y1, x2, y2 = gt_box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='red', facecolor='none', alpha=0.8)
                ax.add_patch(rect)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='cyan', lw=4, alpha=0.6, label=f'Ground Truth ({len(gt_boxes)})'),
            Line2D([0], [0], color='green', lw=2, label=f'True Positives ({tp_count})'),
            Line2D([0], [0], color='orange', lw=2, label=f'False Positives ({fp_count})'),
            Line2D([0], [0], color='red', lw=2, label=f'False Negatives ({fn_count})')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f'Detection Results: TP={tp_count}, FP={fp_count}, FN={fn_count}', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        return fig


class AntDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Forest Ant Detection System")
        self.root.geometry("1600x1000")
        
        # Try to maximize window (works on most systems)
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass  # macOS or other
        
        # Initialize detector
        self.detector = RandomForestAntDetector()
        
        # Data storage
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.training_data = []
        self.test_data = []
        
        # UI state
        self.drawing = False
        self.rect_start = None
        self.current_rect = None
        
        # Zoom variables
        self.zoom_factor = 1.0
        
        # Prediction navigation
        self.prediction_images = []
        self.pred_index = 0
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        self.update_memory_usage()
        
    def setup_styles(self):
        """Configure ttk styles with dark blue and gold theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Professional color palette
        self.colors = {
            'bg': '#1a2332',              # Dark blue background
            'bg_light': '#2d3e50',        # Lighter blue for frames
            'fg': '#ffffff',              # White text
            'accent': '#d4af37',          # Gold accent
            'accent_hover': '#f4cf57',    # Lighter gold for hover
            'secondary': '#4a90e2',       # Light blue secondary
            'warning': '#ff9800',         # Orange warning
            'error': '#e74c3c',           # Red error
            'success': '#27ae60',         # Green success
            'canvas_bg': '#ffffff',       # White canvas
            'button_text': '#ffffff'      # White button text
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg'])
        
        # Frame styles
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabelframe', background=self.colors['bg'], foreground=self.colors['fg'])
        style.configure('TLabelframe.Label', background=self.colors['bg'], foreground=self.colors['accent'], 
                       font=('Arial', 10, 'bold'))
        
        # Label styles
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'], 
                       font=('Arial', 9))
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), foreground=self.colors['accent'])
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), foreground=self.colors['accent'])
        style.configure('Info.TLabel', font=('Arial', 9), foreground=self.colors['fg'])
        style.configure('Success.TLabel', foreground=self.colors['success'], font=('Arial', 9, 'bold'))
        style.configure('Error.TLabel', foreground=self.colors['error'], font=('Arial', 9, 'bold'))
        
        # Button styles - Gold buttons with white text
        style.configure('TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['button_text'],
                       borderwidth=1,
                       focuscolor='none',
                       font=('Arial', 9, 'bold'),
                       padding=(10, 5))
        
        style.map('TButton',
                 background=[('active', self.colors['accent_hover']), 
                            ('pressed', self.colors['accent'])],
                 foreground=[('active', self.colors['button_text']),
                            ('pressed', self.colors['button_text'])])
        
        # Notebook (tabs) styles
        style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=self.colors['bg_light'],
                       foreground=self.colors['fg'],
                       padding=[20, 10],
                       font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['accent'])],
                 foreground=[('selected', self.colors['button_text'])],
                 expand=[('selected', [1, 1, 1, 0])])
        
        # Progressbar style
        style.configure('TProgressbar',
                       background=self.colors['accent'],
                       troughcolor=self.colors['bg_light'],
                       borderwidth=0,
                       thickness=20)
        
        # Checkbutton style
        style.configure('TCheckbutton',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       font=('Arial', 9))
        style.map('TCheckbutton',
                 background=[('active', self.colors['bg'])],
                 foreground=[('active', self.colors['accent'])])
        
        # Combobox style
        style.configure('TCombobox',
                       fieldbackground=self.colors['bg_light'],
                       background=self.colors['accent'],
                       foreground=self.colors['fg'],
                       arrowcolor=self.colors['accent'])
        
        # Spinbox style
        style.configure('TSpinbox',
                       fieldbackground=self.colors['bg_light'],
                       background=self.colors['accent'],
                       foreground=self.colors['fg'],
                       arrowcolor=self.colors['accent'])
        
        # Scrollbar style
        style.configure('TScrollbar',
                       background=self.colors['accent'],
                       troughcolor=self.colors['bg_light'],
                       borderwidth=0,
                       arrowcolor=self.colors['button_text'])
        style.map('TScrollbar',
                 background=[('active', self.colors['accent_hover'])])
        
        # Separator style
        style.configure('TSeparator', background=self.colors['accent'])
        
    def create_widgets(self):
        """Create main UI widgets with enhanced styling"""
        # Main container with notebook (tabs)
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Title frame with gradient effect
        title_frame = ttk.Frame(main_container)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title with icon effect
        title_text = "🐜 Random Forest Ant Detection System"
        title = ttk.Label(title_frame, text=title_text, style='Title.TLabel')
        title.pack(side=tk.LEFT, padx=10)
        
        # Memory and status info with better styling
        info_container = ttk.Frame(title_frame)
        info_container.pack(side=tk.RIGHT, padx=10)
        
        self.memory_label = ttk.Label(info_container, text="💾 Memory: -- MB", style='Info.TLabel')
        self.memory_label.pack(side=tk.RIGHT, padx=10)
        
        # Create notebook for different modes
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_annotation_tab()
        self.create_training_tab()
        self.create_testing_tab()
        self.create_prediction_tab()
        
        # Status bar with better styling
        self.status_frame = ttk.Frame(main_container)
        self.status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label with icon
        self.status_label = ttk.Label(self.status_frame, text="✓ Ready", style='Info.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Progress bar with better visibility
        self.progress = ttk.Progressbar(self.status_frame, mode='determinate', length=400)
        self.progress.pack(side=tk.RIGHT, padx=10)
    
    def create_annotation_tab(self):
        """Create annotation tab with zoom functionality"""
        annotation_frame = ttk.Frame(self.notebook)
        self.notebook.add(annotation_frame, text="Annotation")
        
        # Configure grid
        annotation_frame.columnconfigure(1, weight=1)
        annotation_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(annotation_frame, text="Annotation Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        control_frame.rowconfigure(10, weight=1)
        
        # Load image button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Zoom controls
        ttk.Label(control_frame, text="Zoom Controls:", style='Heading.TLabel').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom, width=10).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(control_frame, text="Zoom: 100%", style='Info.TLabel')
        self.zoom_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Annotation tools
        ttk.Label(control_frame, text="Annotation Tools:", style='Heading.TLabel').grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        ttk.Button(control_frame, text="Clear All Annotations", command=self.clear_annotations).grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(control_frame, text="Save Annotations", command=self.save_annotations_auto).grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Current image info
        ttk.Label(control_frame, text="Current Image:", style='Heading.TLabel').grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        self.image_info_label = ttk.Label(control_frame, text="No image loaded", style='Info.TLabel')
        self.image_info_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Annotations list
        ttk.Label(control_frame, text="Annotations:", style='Heading.TLabel').grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(control_frame)
        list_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        scrollbar_ann = ttk.Scrollbar(list_frame)
        scrollbar_ann.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.annotation_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar_ann.set, height=15,
                                            bg=self.colors['bg_light'], fg=self.colors['fg'],
                                            selectbackground=self.colors['accent'], 
                                            selectforeground=self.colors['button_text'],
                                            font=('Arial', 9), relief=tk.FLAT)
        self.annotation_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_ann.config(command=self.annotation_listbox.yview)
        
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_annotation).grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        instructions = """
Instructions:
- Click and drag to draw rectangles around ants
- Use zoom controls to see details clearly
- Mouse wheel can also zoom in/out
- Annotations auto-save with image name
- Clear all to start over
"""
        instructions_label = ttk.Label(control_frame, text=instructions, style='Info.TLabel', justify=tk.LEFT)
        instructions_label.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Right panel - Canvas with scrollbars
        self.create_zoomable_canvas(annotation_frame, 0, 1)
    
    def create_zoomable_canvas(self, parent, row, column):
        """Create canvas with zoom and scroll capabilities"""
        canvas_frame = ttk.LabelFrame(parent, text="Image View", padding="5")
        canvas_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create canvas container
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        # Scrollbars
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_container, bg=self.colors['canvas_bg'],
                               xscrollcommand=self.h_scroll.set, 
                               yscrollcommand=self.v_scroll.set,
                               highlightthickness=2, highlightbackground=self.colors['accent'])
        
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
        self.canvas.focus_set()
    
    def create_training_tab(self):
        """Create training tab with all original parameters"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training")
        
        # Configure grid
        training_frame.columnconfigure(1, weight=1)
        training_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(training_frame, text="Training Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        control_frame.rowconfigure(15, weight=1)
        
        # Data loading
        ttk.Label(control_frame, text="Data Loading:", style='Heading.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Button(control_frame, text="Load Training Images (Batch)", 
                  command=self.load_training_images_batch).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(control_frame, text="Load Training Annotations (Batch)", 
                  command=self.load_training_annotations_batch).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Training data info
        self.training_info_label = ttk.Label(control_frame, text="No training data loaded", style='Info.TLabel')
        self.training_info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Button(control_frame, text="View Training Pairs", command=self.view_training_pairs).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Detection parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(control_frame, text="Detection Parameters:", style='Heading.TLabel').grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Patch size
        ttk.Label(control_frame, text="Patch Size:").grid(row=7, column=0, sticky=tk.W)
        self.patch_size_var = tk.StringVar(value="32x32")
        patch_combo = ttk.Combobox(control_frame, textvariable=self.patch_size_var, 
                                  values=["16x16", "24x24", "32x32", "48x48"], state="readonly", width=10)
        patch_combo.grid(row=7, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Stride
        ttk.Label(control_frame, text="Detection Stride:").grid(row=8, column=0, sticky=tk.W)
        self.stride_var = tk.IntVar(value=16)
        ttk.Spinbox(control_frame, from_=4, to=32, textvariable=self.stride_var, width=10).grid(row=8, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Scales
        ttk.Label(control_frame, text="Detection Scales:").grid(row=9, column=0, sticky=tk.W)
        scales_frame = ttk.Frame(control_frame)
        scales_frame.grid(row=9, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.scale_vars = {}
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for i, scale in enumerate(scales):
            var = tk.BooleanVar(value=scale in [0.75, 1.0, 1.25])
            self.scale_vars[scale] = var
            ttk.Checkbutton(scales_frame, text=str(scale), variable=var).grid(row=i//3, column=i%3, sticky=tk.W)
        
        # Distance threshold with labels
        ttk.Label(control_frame, text="Distance Threshold:").grid(row=10, column=0, sticky=tk.W)
        
        # Create frame for slider and labels
        distance_frame = ttk.Frame(control_frame)
        distance_frame.grid(row=10, column=1, sticky=(tk.W, tk.E), padx=5)
        distance_frame.columnconfigure(1, weight=1)
        
        # Min label
        ttk.Label(distance_frame, text="0.005", style='Info.TLabel').grid(row=0, column=0)
        
        # Slider
        self.distance_threshold_var = tk.DoubleVar(value=0.05)
        distance_scale = tk.Scale(distance_frame, from_=0.005, to=0.20, 
                                  variable=self.distance_threshold_var, orient=tk.HORIZONTAL, resolution=0.005,
                                  bg=self.colors['bg'], fg=self.colors['fg'], 
                                  troughcolor=self.colors['bg_light'],
                                  activebackground=self.colors['accent'],
                                  highlightthickness=0, sliderrelief=tk.FLAT)
        distance_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Max label
        ttk.Label(distance_frame, text="0.20", style='Info.TLabel').grid(row=0, column=2)
        
        # Current value label
        self.distance_value_label = ttk.Label(control_frame, text="Current: 0.05", style='Info.TLabel')
        self.distance_value_label.grid(row=11, column=1, sticky=tk.W, padx=5)
        
        # Update label when slider changes
        def update_distance_label(*args):
            value = self.distance_threshold_var.get()
            self.distance_value_label.config(text=f"Current: {value:.3f}")
        
        self.distance_threshold_var.trace('w', update_distance_label)
        
        # Random Forest parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(control_frame, text="Model Parameters:", style='Heading.TLabel').grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.tune_hyperparams_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto-tune Hyperparameters (Recommended)", 
                       variable=self.tune_hyperparams_var).grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Info about hyperparameter tuning
        tuning_info = """Hyperparameter tuning will automatically optimize:
- Number of trees (25-100)
- Maximum depth (6-15)
- Minimum samples for split/leaf
- Feature selection method"""
        
        info_label = ttk.Label(control_frame, text=tuning_info, style='Info.TLabel', justify=tk.LEFT)
        info_label.grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Training controls
        ttk.Button(control_frame, text="Train Model", command=self.train_model).grid(row=16, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Save/Load model with report download
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=17, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(0, weight=1)
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 0))
        
        # Download report button
        self.download_report_btn = ttk.Button(control_frame, text="Download Training Report", 
                                             command=self.download_training_report, state='disabled')
        self.download_report_btn.grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Right panel - Results
        self.create_training_results_panel(training_frame, 0, 1)
    
    def create_training_results_panel(self, parent, row, column):
        """Create training results panel"""
        results_frame = ttk.LabelFrame(parent, text="Training Results", padding="5")
        results_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Metrics display
        metrics_frame = ttk.Frame(results_frame)
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create metric labels
        self.training_metrics = {}
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Samples', 'Features']
        
        for i, name in enumerate(metric_names):
            label = ttk.Label(metrics_frame, text=f"{name}:", style='Info.TLabel')
            label.grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5)
            
            value_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
            value_label.grid(row=i//3, column=(i%3)*2+1, sticky=tk.W, padx=5)
            self.training_metrics[name.lower()] = value_label
        
        # Training log
        log_frame = ttk.LabelFrame(results_frame, text="Training Log")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        scroll_training = ttk.Scrollbar(log_frame)
        scroll_training.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.training_log = tk.Text(log_frame, wrap=tk.WORD, yscrollcommand=scroll_training.set, height=20,
                                   bg=self.colors['bg_light'], fg=self.colors['fg'], 
                                   insertbackground=self.colors['accent'],
                                   font=('Consolas', 9), relief=tk.FLAT, padx=10, pady=10)
        self.training_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scroll_training.config(command=self.training_log.yview)
    
    def create_test_results_panel(self, parent, row, column):
        """Create test results panel"""
        results_frame = ttk.LabelFrame(parent, text="Evaluation Results", padding="5")
        results_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Metrics display
        metrics_frame = ttk.Frame(results_frame)
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create test metric labels
        self.test_metrics = {}
        metric_names = ['Precision', 'Recall', 'F1-Score', 'mAP', 'True Positives', 'False Positives', 'False Negatives', 'Kappa', 'Detection Accuracy']
        
        for i, name in enumerate(metric_names):
            label = ttk.Label(metrics_frame, text=f"{name}:", style='Info.TLabel')
            label.grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5)
            
            value_label = ttk.Label(metrics_frame, text="--", style='Info.TLabel')
            value_label.grid(row=i//3, column=(i%3)*2+1, sticky=tk.W, padx=5)
            self.test_metrics[name.lower().replace(' ', '_')] = value_label
        
        # Results log
        log_frame = ttk.LabelFrame(results_frame, text="Detailed Results")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        scroll_test = ttk.Scrollbar(log_frame)
        scroll_test.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.test_results_text = tk.Text(log_frame, wrap=tk.WORD, yscrollcommand=scroll_test.set,
                                        bg=self.colors['bg_light'], fg=self.colors['fg'], 
                                        insertbackground=self.colors['accent'],
                                        font=('Consolas', 9), relief=tk.FLAT, padx=10, pady=10)
        self.test_results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scroll_test.config(command=self.test_results_text.yview)
    
    def create_prediction_tab(self):
        """Create the prediction/detection tab - COMPLETE VERSION with all controls"""
        pred_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pred_frame, text="🔍 Detection")
        
        # Configure grid weights for side-by-side layout
        pred_frame.columnconfigure(0, weight=0)  # Control panel (fixed width)
        pred_frame.columnconfigure(1, weight=1)  # Results panel (expandable)
        pred_frame.rowconfigure(0, weight=1)
        
        # Left side: Control Panel
        control_panel = ttk.LabelFrame(pred_frame, text="Detection Controls", padding="10")
        control_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        current_row = 0
        
        # 1. MODEL LOADING SECTION
        ttk.Label(control_panel, text="1. Load Model", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        ttk.Button(control_panel, text="📁 Load Trained Model", 
                  command=self.load_model_for_prediction).grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        self.pred_model_status_label = ttk.Label(control_panel, text="No model loaded", style='Info.TLabel')
        self.pred_model_status_label.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        ttk.Separator(control_panel, orient='horizontal').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # 2. LOAD IMAGES SECTION
        ttk.Label(control_panel, text="2. Load Images", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        ttk.Button(control_panel, text="📁 Select Images", 
                  command=self.load_prediction_images_batch).grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        self.pred_info_label = ttk.Label(control_panel, text="No images loaded", 
                                         style='Info.TLabel', wraplength=200)
        self.pred_info_label.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        ttk.Separator(control_panel, orient='horizontal').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # 3. DETECTION PARAMETERS SECTION
        ttk.Label(control_panel, text="3. Detection Parameters", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        # Confidence Threshold
        conf_frame = ttk.Frame(control_panel)
        conf_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        conf_frame.columnconfigure(1, weight=1)
        
        ttk.Label(conf_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        
        self.pred_confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, 
                                   variable=self.pred_confidence_var, orient=tk.HORIZONTAL, resolution=0.01,
                                   bg=self.colors['bg'], fg=self.colors['fg'], 
                                   troughcolor=self.colors['bg_light'],
                                   activebackground=self.colors['accent'],
                                   highlightthickness=0, sliderrelief=tk.FLAT)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.pred_confidence_value_label = ttk.Label(conf_frame, text="0.600", style='Info.TLabel')
        self.pred_confidence_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        
        def update_pred_confidence_label(*args):
            value = self.pred_confidence_var.get()
            self.pred_confidence_value_label.config(text=f"{value:.3f}")
        
        self.pred_confidence_var.trace('w', update_pred_confidence_label)
        current_row += 1
        
        # Distance evaluation method
        ttk.Label(control_panel, text="Distance Method:", style='Info.TLabel').grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))
        current_row += 1
        
        self.pred_normalized_var = tk.BooleanVar(value=True)
        normalized_check = ttk.Checkbutton(control_panel, text="Use Normalized Distance", 
                                          variable=self.pred_normalized_var,
                                          command=self.update_pred_distance_ui)
        normalized_check.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Distance threshold frame
        self.pred_distance_frame = ttk.Frame(control_panel)
        self.pred_distance_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.pred_distance_frame.columnconfigure(1, weight=1)
        current_row += 1
        
        # Initialize distance threshold
        self.pred_distance_var = tk.DoubleVar(value=0.05)
        self.setup_pred_distance_ui()
        
        # Fast detection mode
        self.pred_fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_panel, text="Fast Detection Mode", 
                       variable=self.pred_fast_mode_var).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Show all detections
        self.show_all_detections_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_panel, text="Show All Detections (Before NMS)", 
                       variable=self.show_all_detections_var).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Detection color
        color_frame = ttk.Frame(control_panel)
        color_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(color_frame, text="Box Color:").pack(side=tk.LEFT)
        
        self.detection_color_var = tk.StringVar(value="#00FF00")  # Green default
        ttk.Button(color_frame, text="Choose Color", command=self.choose_detection_color).pack(
            side=tk.LEFT, padx=5)
        current_row += 1
        
        ttk.Separator(control_panel, orient='horizontal').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # 4. RUN DETECTION SECTION
        ttk.Label(control_panel, text="4. Run Detection", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        ttk.Button(control_panel, text="▶️ Start Detection", 
                  command=self.run_batch_detection,
                  style='Accent.TButton').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        ttk.Separator(control_panel, orient='horizontal').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # 5. NAVIGATE RESULTS SECTION
        ttk.Label(control_panel, text="5. Navigate Results", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        nav_frame = ttk.Frame(control_panel)
        nav_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(nav_frame, text="◀ Previous", command=self.prev_prediction, width=10).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_prediction, width=10).pack(
            side=tk.RIGHT, padx=2)
        current_row += 1
        
        self.pred_index_label = ttk.Label(control_panel, text="0/0", style='Info.TLabel')
        self.pred_index_label.grid(row=current_row, column=0, columnspan=2, pady=5)
        current_row += 1
        
        ttk.Separator(control_panel, orient='horizontal').grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # 6. EXPORT SECTION
        ttk.Label(control_panel, text="6. Export Results", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        current_row += 1
        
        # First row: Summary and Detailed CSV side by side
        export_frame1 = ttk.Frame(control_panel)
        export_frame1.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        export_frame1.columnconfigure(0, weight=1)
        export_frame1.columnconfigure(1, weight=1)
        
        self.export_summary_btn = ttk.Button(export_frame1, text="📊 Summary CSV", 
                          command=self.export_detection_summary, state='disabled')
        self.export_summary_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))
        
        self.export_detailed_btn = ttk.Button(export_frame1, text="📋 Detailed CSV", 
                          command=self.export_detection_detailed, state='disabled')
        self.export_detailed_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 0))
        current_row += 1
        
        # Second row: Download Image button
        ttk.Button(control_panel, text="📥 Download Image", 
                  command=self.download_predicted_image).grid(
            row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        current_row += 1
        
        # Add expandable space at the end
        current_row += 1
        control_panel.rowconfigure(current_row, weight=1)
        
        # Right side: Results Panel (keep existing code)
        results_panel = ttk.Frame(pred_frame)
        results_panel.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        results_panel.columnconfigure(0, weight=1)
        results_panel.rowconfigure(1, weight=1)
        
        # Batch summary at top
        summary_frame = ttk.LabelFrame(results_panel, text="Batch Summary", padding="5")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.pred_summary_metrics = {}
        summary_names = ['Images Processed', 'Total Ants Found', 'Average per Image', 'Processing Time']
        
        for i, name in enumerate(summary_names):
            row_pos = i // 2
            col_pos = (i % 2) * 2
            
            label = ttk.Label(summary_frame, text=f"{name}:", style='Info.TLabel')
            label.grid(row=row_pos, column=col_pos, sticky=tk.W, padx=10, pady=2)
            
            value_label = ttk.Label(summary_frame, text="--", style='Info.TLabel', font=('Arial', 9, 'bold'))
            value_label.grid(row=row_pos, column=col_pos+1, sticky=tk.W, padx=5, pady=2)
            self.pred_summary_metrics[name.lower().replace(' ', '_')] = value_label
        
        # Image display frame
        display_frame = ttk.LabelFrame(results_panel, text="Current Detection Result", padding="5")
        display_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(2, weight=1)
        
        # Current image info
        info_frame = ttk.Frame(display_frame)
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.current_image_label = ttk.Label(info_frame, text="No image selected", style='Info.TLabel')
        self.current_image_label.pack(side=tk.LEFT)
        
        self.detection_count_label = ttk.Label(info_frame, text="Detections: 0", style='Info.TLabel')
        self.detection_count_label.pack(side=tk.RIGHT)
        
        # Zoom controls
        zoom_frame = ttk.Frame(display_frame)
        zoom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(zoom_frame, text="Zoom:", style='Info.TLabel').pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="🔍+", command=self.zoom_prediction_in, width=5).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", style='Info.TLabel', width=8)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="🔍-", command=self.zoom_prediction_out, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="↺ Reset", command=self.zoom_prediction_reset, width=8).pack(side=tk.LEFT, padx=5)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(display_frame)
        canvas_container.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        
        self.pred_canvas = tk.Canvas(canvas_container, bg='white',
                                   xscrollcommand=h_scroll.set,
                                   yscrollcommand=v_scroll.set,
                                   highlightthickness=2, 
                                   highlightbackground=self.colors['accent'])
        
        h_scroll.config(command=self.pred_canvas.xview)
        v_scroll.config(command=self.pred_canvas.yview)
        
        self.pred_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        h_scroll.grid(row=1, column=0, sticky=(tk.E, tk.W))
        v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initialize zoom level
        self.pred_zoom_level = 1.0
        
        # Initialize display
        self.pred_canvas.create_text(200, 100, text="Load images and run detection to see results", 
                                   fill="gray", font=("Arial", 12))



    def load_model_for_prediction(self):
        """Load trained model from file for prediction"""
        filepath = filedialog.askopenfilename(
            title="Load Model for Prediction",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.detector.load_model_from_file(filepath):
                    messagebox.showinfo("Success", f"Model loaded from {filepath}")
                    self.update_status(f"Model loaded: {os.path.basename(filepath)}")
                    
                    # Update model status labels
                    if hasattr(self, 'pred_model_status_label'):
                        self.pred_model_status_label.config(text="Model loaded ✓", style='Success.TLabel')
                    return True
                else:
                    messagebox.showerror("Error", "Failed to load model")
                    return False
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                return False
        return False


    def download_predicted_image(self):
        """Download the current predicted image with annotations at full resolution"""
        if not hasattr(self, 'current_pred_image_full_res') or self.current_pred_image_full_res is None:
            messagebox.showwarning("Warning", "No predicted image to download. Please run detection first.")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ant_detection_annotated_{timestamp}.png"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            title="Save Annotated Image",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Save with high quality
                if filepath.lower().endswith('.png'):
                    # PNG: lossless, use maximum compression
                    self.current_pred_image_full_res.save(filepath, 'PNG', compress_level=9)
                elif filepath.lower().endswith(('.jpg', '.jpeg')):
                    # JPEG: save with high quality
                    self.current_pred_image_full_res.save(filepath, 'JPEG', quality=95, subsampling=0)
                else:
                    # Default to PNG
                    self.current_pred_image_full_res.save(filepath, 'PNG', compress_level=9)
                
                # Get image dimensions for info message
                width, height = self.current_pred_image_full_res.size
                messagebox.showinfo("Download Successful", 
                    f"Full-resolution annotated image saved!\n\n"
                    f"File: {filepath}\n"
                    f"Resolution: {width} x {height} pixels")
            except Exception as e:
                messagebox.showerror("Download Error", f"Error saving image: {str(e)}")
    
    def zoom_prediction_in(self):
        """Zoom in on the prediction canvas"""
        if hasattr(self, 'pred_zoom_level'):
            self.pred_zoom_level = min(self.pred_zoom_level * 1.2, 5.0)  # Max 5x zoom
            self.update_prediction_display()
    
    def zoom_prediction_out(self):
        """Zoom out on the prediction canvas"""
        if hasattr(self, 'pred_zoom_level'):
            self.pred_zoom_level = max(self.pred_zoom_level / 1.2, 0.5)  # Min 0.5x zoom
            self.update_prediction_display()
    
    def zoom_prediction_reset(self):
        """Reset zoom to 100%"""
        if hasattr(self, 'pred_zoom_level'):
            self.pred_zoom_level = 1.0
            self.update_prediction_display()
    
    def update_prediction_display(self):
        """Update the prediction canvas with current zoom level"""
        # Use full resolution image if available, otherwise fall back to display image
        if hasattr(self, 'current_pred_image_full_res') and self.current_pred_image_full_res is not None:
            source_image = self.current_pred_image_full_res
        elif hasattr(self, 'current_pred_image_pil') and self.current_pred_image_pil is not None:
            source_image = self.current_pred_image_pil
        else:
            return
        
        # Calculate new size based on zoom
        # For full res image, scale down first then apply zoom
        original_width, original_height = source_image.size
        
        # If using full res, calculate base display size first
        if hasattr(self, 'current_pred_image_full_res') and source_image == self.current_pred_image_full_res:
            max_display_height = 400
            if original_height > max_display_height:
                aspect_ratio = original_width / original_height
                base_width = int(max_display_height * aspect_ratio)
                base_height = max_display_height
            else:
                base_width = original_width
                base_height = original_height
        else:
            base_width = original_width
            base_height = original_height
        
        # Apply zoom to base size
        new_width = int(base_width * self.pred_zoom_level)
        new_height = int(base_height * self.pred_zoom_level)
        
        # Resize image from full resolution source for better quality
        resized_image = source_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.current_pred_photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        self.pred_canvas.delete("all")
        self.pred_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_pred_photo)
        self.pred_canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.pred_zoom_level * 100)}%")
    
  
    # Event handlers and functionality
    def load_image(self):
        """Load a single image for annotation"""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Load image
                image = cv2.imread(filepath)
                if image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image
                self.current_image_path = filepath
                self.detector.original_filename = os.path.basename(filepath)
                
                # Update UI
                self.display_image_on_canvas(image)
                self.update_image_info()
                self.clear_annotations()
                self.update_status(f"Loaded: {os.path.basename(filepath)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def display_image_on_canvas(self, image):
        """Display image on canvas with zoom support"""
        if image is None:
            return
        
        # Calculate display size based on zoom
        h, w = image.shape[:2]
        display_w = int(w * self.zoom_factor)
        display_h = int(h * self.zoom_factor)
        
        # Resize image for display
        if self.zoom_factor != 1.0:
            display_image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_image = image.copy()
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(display_image)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        
        # Clear canvas and display
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image, tags="image")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Redraw annotations
        self.redraw_annotations()
        
        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
    
    def update_image_info(self):
        """Update image information display"""
        if self.current_image is not None:
            h, w, c = self.current_image.shape
            filename = os.path.basename(self.current_image_path) if self.current_image_path else "Unknown"
            info_text = f"{filename} - {w}x{h} pixels - {len(self.annotations)} annotations"
            self.image_info_label.config(text=info_text)
        else:
            self.image_info_label.config(text="No image loaded")
    
    # Zoom functionality
    def zoom_in(self):
        """Zoom in on image"""
        if self.current_image is not None:
            self.zoom_factor = min(self.zoom_factor * 1.25, 5.0)
            self.display_image_on_canvas(self.current_image)
    
    def zoom_out(self):
        """Zoom out on image"""
        if self.current_image is not None:
            self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)
            self.display_image_on_canvas(self.current_image)
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.current_image is not None:
            self.zoom_factor = 1.0
            self.display_image_on_canvas(self.current_image)
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_image is None:
            return
        
        # Determine zoom direction
        if hasattr(event, 'delta'):
            # Windows
            if event.delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, 5.0)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
        else:
            # Linux
            if event.num == 4:
                self.zoom_factor = min(self.zoom_factor * 1.1, 5.0)
            elif event.num == 5:
                self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
        
        self.display_image_on_canvas(self.current_image)
    
    # Annotation functionality
    def on_canvas_click(self, event):
        """Handle canvas click for annotation"""
        if self.current_image is None:
            return
        
        self.drawing = True
        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Adjust for zoom
        img_x = canvas_x / self.zoom_factor
        img_y = canvas_y / self.zoom_factor
        
        self.rect_start = (img_x, img_y)
        self.canvas_start = (canvas_x, canvas_y)
        
        # Create rectangle on canvas (at zoom level)
        self.current_rect = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline='green', width=2, tags="current_annotation"
        )
    
    def on_canvas_drag(self, event):
        """Handle canvas drag for annotation"""
        if not self.drawing or not self.current_rect:
            return
        
        # Convert to canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Update rectangle on canvas
        self.canvas.coords(self.current_rect, 
                          self.canvas_start[0], self.canvas_start[1], canvas_x, canvas_y)
    
    def on_canvas_release(self, event):
        """Handle canvas release for annotation"""
        if not self.drawing or not self.current_rect:
            return
        
        self.drawing = False
        
        # Convert final coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        img_x = canvas_x / self.zoom_factor
        img_y = canvas_y / self.zoom_factor
        
        # Calculate annotation bounds in image coordinates
        x1 = min(self.rect_start[0], img_x)
        y1 = min(self.rect_start[1], img_y)
        x2 = max(self.rect_start[0], img_x)
        y2 = max(self.rect_start[1], img_y)
        
        # Check if annotation is large enough
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            # Add to annotations
            annotation = {
                'x': x1,
                'y': y1, 
                'width': x2 - x1,
                'height': y2 - y1
            }
            self.annotations.append(annotation)
            self.update_annotation_list()
            self.update_image_info()
            
            # Convert back to display coordinates and redraw properly
            self.redraw_annotations()
        else:
            # Remove if too small
            self.canvas.delete(self.current_rect)
        
        self.current_rect = None
    
    def redraw_annotations(self):
        """Redraw all annotations on canvas"""
        # Clear existing annotation drawings
        self.canvas.delete("annotation")
        
        # Draw each annotation at current zoom level
        for i, ann in enumerate(self.annotations):
            x1 = ann['x'] * self.zoom_factor
            y1 = ann['y'] * self.zoom_factor
            x2 = (ann['x'] + ann['width']) * self.zoom_factor
            y2 = (ann['y'] + ann['height']) * self.zoom_factor
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='green', width=2, tags="annotation"
            )
            
            # Add label - just the number in smaller font directly above left corner
            label_text = str(i+1)
            self.canvas.create_text(
                x1, y1 - 2, text=label_text,
                fill='green', anchor=tk.SW, font=('Arial', 7), tags="annotation"
            )
    
    def update_annotation_list(self):
        """Update annotation listbox"""
        self.annotation_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            text = f"Ant {i+1}: ({int(ann['x'])}, {int(ann['y'])}) {int(ann['width'])}x{int(ann['height'])} px"
            self.annotation_listbox.insert(tk.END, text)
    
    def delete_annotation(self):
        """Delete selected annotation"""
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.annotations[index]
            self.update_annotation_list()
            self.update_image_info()
            # Redraw the entire image with remaining annotations
            self.display_image_on_canvas(self.current_image)
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []
        self.canvas.delete("annotation")
        self.annotation_listbox.delete(0, tk.END)
        self.update_image_info()
    
    def save_annotations_auto(self):
        """Automatically save annotations with image name + _annotations.json"""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save!")
            return
    
        if not self.current_image_path:
            messagebox.showerror("Error", "No image loaded!")
            return
    
        try:
            # Generate filename automatically
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            filename = f"{base_name}_annotations.json"
        
            # Get the directory of the current image
            image_dir = os.path.dirname(self.current_image_path)
            filepath = os.path.join(image_dir, filename)
        
            # Prepare annotation data in the exact format from your example
            data = {
                'image_path': os.path.basename(self.current_image_path),
                'annotations': []
            }
        
            # Convert annotations to match your format exactly
            for ann in self.annotations:
                annotation_data = {
                    'x': float(ann['x']),
                    'y': float(ann['y']),
                    'width': float(ann['width']),
                    'height': float(ann['height'])
                }
                data['annotations'].append(annotation_data)
        
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
            messagebox.showinfo("Success", f"Annotations saved to {filename}")
            self.update_status(f"Annotations saved: {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error saving annotations: {str(e)}")
    
    def load_training_images_batch(self):
        """Load multiple training images"""
        filepaths = filedialog.askopenfilenames(
            title="Select Training Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filepaths:
            training_images = []
            failed_images = []
            
            for filepath in filepaths:
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        training_images.append({
                            'image': image,
                            'path': filepath,
                            'filename': os.path.basename(filepath)
                        })
                    else:
                        failed_images.append(filepath)
                except Exception as e:
                    failed_images.append(filepath)
            
            if training_images:
                # Store in instance variable
                if not hasattr(self, 'training_images'):
                    self.training_images = []
                self.training_images.extend(training_images)
                
                self.update_training_info()
                messagebox.showinfo("Success", f"Loaded {len(training_images)} training images")
            
            if failed_images:
                messagebox.showwarning("Warning", f"Failed to load {len(failed_images)} images")
    

    def load_training_annotations_batch(self):
        """Load multiple annotation files"""
        filepaths = filedialog.askopenfilenames(
            title="Select Annotation Files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepaths:
            training_annotations = []
            failed_annotations = []
            
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Extract image name and annotations
                    image_path = data.get('image_path', os.path.basename(filepath))
                    annotations = data.get('annotations', [])
                    
                    training_annotations.append({
                        'image_path': image_path,
                        'annotations': annotations,
                        'annotation_file': filepath
                    })
                    
                except Exception as e:
                    failed_annotations.append(filepath)
            
            if training_annotations:
                # Store in instance variable
                if not hasattr(self, 'training_annotations'):
                    self.training_annotations = []
                self.training_annotations.extend(training_annotations)
                
                self.update_training_info()
                messagebox.showinfo("Success", f"Loaded {len(training_annotations)} annotation files")
            
            if failed_annotations:
                messagebox.showwarning("Warning", f"Failed to load {len(failed_annotations)} annotation files")


    def update_training_info(self):
        """Update training info display"""
        num_images = len(getattr(self, 'training_images', []))
        num_annotations = len(getattr(self, 'training_annotations', []))
        
        info_text = f"Images: {num_images}, Annotations: {num_annotations}"
        if hasattr(self, 'training_info_label'):
            self.training_info_label.config(text=info_text)


    def view_training_pairs(self):
        """View training image-annotation pairs with pagination"""
        if not hasattr(self, 'training_images') or not hasattr(self, 'training_annotations'):
            messagebox.showwarning("Warning", "Please load training images and annotations first")
            return
        
        # Match images with annotations
        matched_pairs = []
        for img_data in self.training_images:
            img_filename = img_data['filename']
            for ann_data in self.training_annotations:
                ann_image_path = ann_data['image_path']
                if os.path.basename(ann_image_path) == img_filename:
                    matched_pairs.append({
                        'image': img_data,
                        'annotations': ann_data['annotations'],
                        'annotation_file': ann_data['annotation_file']
                    })
                    break
        
        if not matched_pairs:
            messagebox.showinfo("Info", "No matched pairs found")
            return
        
        # Create window
        pairs_window = tk.Toplevel(self.root)
        pairs_window.title(f"Training Pairs - {len(matched_pairs)} total")
        pairs_window.geometry("1400x900")
        
        try:
            pairs_window.state('zoomed')
        except:
            try:
                pairs_window.attributes('-zoomed', True)
            except:
                pass
        
        # Variables for pagination
        pairs_per_page = 10  # Show 10 pairs at a time
        current_page = [0]  # Use list so it's mutable in nested function
        total_pages = (len(matched_pairs) + pairs_per_page - 1) // pairs_per_page
        
        # Main container
        main_container = ttk.Frame(pairs_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Navigation frame at top
        nav_frame = ttk.Frame(main_container)
        nav_frame.pack(fill=tk.X, pady=5)
        
        page_label = ttk.Label(nav_frame, text=f"Page 1 of {total_pages}", style='Heading.TLabel')
        page_label.pack(side=tk.LEFT, padx=10)
        
        # Content frame with scrollbar
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(content_frame)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def show_page(page_num):
            """Display specific page of pairs"""
            # Clear existing content
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Calculate range
            start_idx = page_num * pairs_per_page
            end_idx = min(start_idx + pairs_per_page, len(matched_pairs))
            
            # Update page label
            page_label.config(text=f"Page {page_num + 1} of {total_pages} | Showing pairs {start_idx + 1}-{end_idx} of {len(matched_pairs)}")
            
            # Display pairs for this page
            for i in range(start_idx, end_idx):
                pair = matched_pairs[i]
                
                pair_frame = ttk.LabelFrame(scrollable_frame, text=f"Pair {i+1}: {pair['image']['filename']}")
                pair_frame.pack(fill=tk.X, padx=5, pady=8)
                
                content_frame_inner = ttk.Frame(pair_frame)
                content_frame_inner.pack(fill=tk.X, padx=15, pady=15)
                
                # Image side
                image_frame = ttk.Frame(content_frame_inner)
                image_frame.pack(side=tk.LEFT, padx=(0, 30))
                
                try:
                    image = pair['image']['image'].copy()
                    h, w = image.shape[:2]
                    
                    for j, ann in enumerate(pair['annotations']):
                        x1, y1 = int(ann['x']), int(ann['y'])
                        x2, y2 = int(ann['x'] + ann['width']), int(ann['y'] + ann['height'])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"{j+1}"
                        cv2.putText(image, label, (x1, max(y1-8, 20)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Original image size from your code
                    display_height = 675
                    aspect_ratio = w / h
                    display_width = int(display_height * aspect_ratio)
                    if display_width > 900:
                        display_width = 900
                        display_height = int(display_width / aspect_ratio)
                    
                    display_image = cv2.resize(image, (display_width, display_height))
                    img_pil = Image.fromarray(display_image)
                    photo = ImageTk.PhotoImage(img_pil)
                    
                    image_label = ttk.Label(image_frame, image=photo)
                    image_label.image = photo  # Keep reference
                    image_label.pack()
                    
                    img_info = ttk.Label(image_frame, text=f"Original Size: {w}x{h} pixels", 
                                       style='Info.TLabel', font=('Arial', 10))
                    img_info.pack(pady=(8, 0))
                    
                except Exception as e:
                    error_label = ttk.Label(image_frame, text=f"Error loading image: {str(e)}", 
                                          style='Error.TLabel')
                    error_label.pack()
                
                # Info side
                info_frame = ttk.Frame(content_frame_inner)
                info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                summary_label = ttk.Label(info_frame, text=f"Annotations: {len(pair['annotations'])}", 
                                        style='Heading.TLabel', font=('Arial', 14, 'bold'))
                summary_label.pack(anchor=tk.W, pady=(0, 15))
                
                details_frame = ttk.Frame(info_frame)
                details_frame.pack(fill=tk.BOTH, expand=True)
                
                details_text = tk.Text(details_frame, height=18, width=60, wrap=tk.WORD, 
                                     font=('Arial', 10),
                                     bg=self.colors['bg_light'], fg=self.colors['fg'], 
                                     insertbackground=self.colors['accent'],
                                     relief=tk.FLAT, padx=10, pady=10)
                details_scroll = ttk.Scrollbar(details_frame, command=details_text.yview)
                details_text.config(yscrollcommand=details_scroll.set)
                
                details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                
                details_text.insert(tk.END, f"Annotation Details for {pair['image']['filename']}\n")
                details_text.insert(tk.END, "=" * 50 + "\n\n")
                
                for j, ann in enumerate(pair['annotations']):
                    details_text.insert(tk.END, f"Ant {j+1}:\n")
                    details_text.insert(tk.END, f"  Position: ({ann['x']:.0f}, {ann['y']:.0f})\n")
                    details_text.insert(tk.END, f"  Size: {ann['width']:.0f} x {ann['height']:.0f} pixels\n")
                    details_text.insert(tk.END, f"  Area: {ann['width'] * ann['height']:.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Center: ({ann['x'] + ann['width']/2:.0f}, {ann['y'] + ann['height']/2:.0f})\n")
                    details_text.insert(tk.END, "\n")
                
                if pair['annotations']:
                    areas = [ann['width'] * ann['height'] for ann in pair['annotations']]
                    details_text.insert(tk.END, "Summary Statistics:\n")
                    details_text.insert(tk.END, f"  Average area: {np.mean(areas):.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Min area: {min(areas):.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Max area: {max(areas):.0f} pxÂ²\n")
                
                details_text.config(state=tk.DISABLED)
            
            canvas.yview_moveto(0)  # Scroll to top
        
        def prev_page():
            if current_page[0] > 0:
                current_page[0] -= 1
                show_page(current_page[0])
        
        def next_page():
            if current_page[0] < total_pages - 1:
                current_page[0] += 1
                show_page(current_page[0])
        
        # Navigation buttons
        ttk.Button(nav_frame, text="< Previous Page", command=prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Page >", command=next_page).pack(side=tk.LEFT, padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Show first page
        show_page(0)
        
        # Mousewheel binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)

    def train_model(self):
        """Train the Random Forest model with comprehensive functionality"""
        if not hasattr(self, 'training_images') or not hasattr(self, 'training_annotations'):
            messagebox.showerror("Error", "Please load training images and annotations first")
            return
        
        # Match images with annotations
        matched_pairs = []
        for img_data in self.training_images:
            img_filename = img_data['filename']
            
            # Find matching annotation
            for ann_data in self.training_annotations:
                ann_image_path = ann_data['image_path']
                if os.path.basename(ann_image_path) == img_filename:
                    matched_pairs.append((img_data['image'], ann_data['annotations']))
                    break
        
        if not matched_pairs:
            messagebox.showerror("Error", "No matching image-annotation pairs found")
            return
        
        # Update detector parameters from UI
        self.update_detector_parameters()
        
        # Get hyperparameter tuning setting
        tune_hyperparams = self.tune_hyperparams_var.get()
        
        # Clear previous training log
        self.training_log.delete(1.0, tk.END)
        
        # Start training in separate thread to avoid freezing UI
        def training_thread():
            try:
                # Update UI - training started
                self.root.after(0, lambda: self.training_log.insert(tk.END, "Starting training...\n"))
                self.root.after(0, lambda: self.training_log.see(tk.END))
                
                # Prepare data
                images = [pair[0] for pair in matched_pairs]
                annotations_list = []
                
                for pair in matched_pairs:
                    # Convert annotations to required format
                    image_annotations = []
                    h, w = pair[0].shape[:2]
                    
                    for ann in pair[1]:
                        # Normalize coordinates
                        x1 = ann['x'] / w
                        y1 = ann['y'] / h
                        x2 = (ann['x'] + ann['width']) / w
                        y2 = (ann['y'] + ann['height']) / h
                        
                        # Clamp to valid range
                        x1 = max(0.0, min(x1, 1.0))
                        y1 = max(0.0, min(y1, 1.0))
                        x2 = max(0.0, min(x2, 1.0))
                        y2 = max(0.0, min(y2, 1.0))
                        
                        if x2 > x1 and y2 > y1:
                            image_annotations.append({
                                'bbox': [x1, y1, x2, y2],
                                'class': 1
                            })
                    
                    annotations_list.append(image_annotations)
                
                # Progress callback
                def progress_callback(current, total, message):
                    progress = int((current / total) * 100)
                    self.root.after(0, lambda: self.progress.configure(value=progress))
                    self.root.after(0, lambda: self.training_log.insert(tk.END, f"{message}\n"))
                    self.root.after(0, lambda: self.training_log.see(tk.END))
                
                # Train model
                results = self.detector.train_model(
                    images, 
                    annotations_list, 
                    tune_hyperparameters=tune_hyperparams,
                    progress_callback=progress_callback
                )
                
                # Update UI with results
                self.root.after(0, lambda: self.update_training_results(results))
                self.root.after(0, lambda: self.progress.configure(value=0))
                
            except Exception as e:
                error_msg = f"Training failed: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))
                self.root.after(0, lambda: self.training_log.insert(tk.END, f"ERROR: {error_msg}\n"))
                self.root.after(0, lambda: self.training_log.see(tk.END))
                self.root.after(0, lambda: self.progress.configure(value=0))
        
        # Start training thread
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    def update_detector_parameters(self):
        """Update detector parameters from UI controls"""
        # Patch size
        patch_size_str = self.patch_size_var.get()
        if patch_size_str == "16x16":
            self.detector.patch_size = (16, 16)
        elif patch_size_str == "24x24":
            self.detector.patch_size = (24, 24)
        elif patch_size_str == "32x32":
            self.detector.patch_size = (32, 32)
        elif patch_size_str == "48x48":
            self.detector.patch_size = (48, 48)
        
        # Stride
        self.detector.stride = self.stride_var.get()
        
        # Scales
        selected_scales = []
        for scale, var in self.scale_vars.items():
            if var.get():
                selected_scales.append(scale)
        if selected_scales:
            self.detector.scales = sorted(selected_scales)
        
        # Distance threshold
        self.detector.distance_threshold = self.distance_threshold_var.get()
        
        # Note: Removed manual RF parameters - always use tuning or defaults

    def download_training_report(self):
        """Download comprehensive training report"""
        # Add debug print to see what's happening
        print("Download training report clicked")
        
        # Check if we have training results
        if not hasattr(self, 'current_training_results'):
            print("No current_training_results attribute found")
            messagebox.showwarning("Warning", "No training results available. Please train a model first.")
            return
        
        if self.current_training_results is None:
            print("current_training_results is None")
            messagebox.showwarning("Warning", "No training results available. Please train a model first.")
            return
        
        print("Training results found, proceeding with download")
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ant_detection_training_report_{timestamp}.txt"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            title="Save Training Report",
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                print(f"Creating report and saving to: {filepath}")
                report_text = self.create_training_report(self.current_training_results)
                
                with open(filepath, 'w') as f:
                    f.write(report_text)
                
                messagebox.showinfo("Success", f"Training report saved to:\n{filepath}")
                self.update_status(f"Report saved: {os.path.basename(filepath)}")
                
            except Exception as e:
                print(f"Error during report generation: {str(e)}")
                messagebox.showerror("Error", f"Error saving report: {str(e)}")
        else:
            print("User cancelled file selection")

    def update_training_results(self, results):
        """Update training results display with download option"""
        try:
            # Store results for download FIRST
            self.current_training_results = results
            
            # Update metric labels
            self.training_metrics['accuracy'].config(text=f"{results['accuracy']:.3f}")
            self.training_metrics['precision'].config(text=f"{results['precision']:.3f}")
            self.training_metrics['recall'].config(text=f"{results['recall']:.3f}")
            self.training_metrics['f1-score'].config(text=f"{results['f1_score']:.3f}")
            self.training_metrics['samples'].config(text=str(results['n_samples']))
            self.training_metrics['features'].config(text=str(results['n_features']))

            # Update training log
            self.training_log.insert(tk.END, "\n" + "="*50 + "\n")
            self.training_log.insert(tk.END, "TRAINING COMPLETED SUCCESSFULLY\n")
            self.training_log.insert(tk.END, "="*50 + "\n")
            self.training_log.insert(tk.END, f"Training Accuracy: {results['accuracy']:.4f}\n")
            self.training_log.insert(tk.END, f"Precision: {results['precision']:.4f}\n")
            self.training_log.insert(tk.END, f"Recall: {results['recall']:.4f}\n")
            self.training_log.insert(tk.END, f"F1-Score: {results['f1_score']:.4f}\n")
            self.training_log.insert(tk.END, f"Training Images: {results['n_samples']}\n")
            self.training_log.insert(tk.END, f"Feature Vector Size: {results['n_features']}\n")
            self.training_log.insert(tk.END, f"Positive Samples: {results['positive_samples']}\n")
            self.training_log.insert(tk.END, f"Negative Samples: {results['negative_samples']}\n")

            if 'best_params' in results and results['best_params']:
                self.training_log.insert(tk.END, f"\nOptimized Parameters:\n")
                for param, value in results['best_params'].items():
                    self.training_log.insert(tk.END, f"  {param}: {value}\n")

                if 'best_score' in results and results['best_score']:
                    self.training_log.insert(tk.END, f"\nBest CV F1-Score: {results['best_score']:.4f}\n")

            # Feature importance summary
            if 'feature_importance' in results:
                self.training_log.insert(tk.END, f"\nTop 10 Most Important Features:\n")
                importance = results['feature_importance']
                top_indices = np.argsort(importance)[-10:][::-1]
                feature_names = getattr(self.detector, 'feature_names', None)
                
                for rank, idx in enumerate(top_indices, 1):
                    if feature_names and idx < len(feature_names):
                        feature_name = feature_names[idx]
                        self.training_log.insert(tk.END, f"  {rank}. {feature_name}: {importance[idx]:.4f}\n")
                    else:
                        self.training_log.insert(tk.END, f"  {rank}. Feature {idx}: {importance[idx]:.4f}\n")

            self.training_log.see(tk.END)

            # Enable the download button - THIS IS THE KEY FIX
            self.download_report_btn.config(state='normal')

            messagebox.showinfo("Success", "Model training completed successfully!")

        except Exception as e:
            self.training_log.insert(tk.END, f"Error updating results: {str(e)}\n")
            self.training_log.see(tk.END)


    def create_training_report(self, results):
        """Create comprehensive training report for download"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_text = f"""Random Forest Ant Detection Model Training Report
====================================================
Generated: {timestamp}

TRAINING CONFIGURATION
----------------------
Patch Size: {self.detector.patch_size}
Detection Stride: {self.detector.stride}
Detection Scales: {self.detector.scales}
Distance Threshold: {self.detector.distance_threshold}
Use Normalized Distance: {self.detector.use_normalized_distance}
Confidence Threshold: {self.detector.confidence_threshold}

CROSS-VALIDATION CONFIGURATION
------------------------------
CV Type: Stratified K-Fold
Number of Folds: Dynamic (2-3 folds based on dataset size)
CV Scoring Metric: F1-Score
Hyperparameter Search: RandomizedSearchCV (6 iterations)
Random State: 42

FEATURE EXTRACTION PARAMETERS
------------------------------
HOG Orientations: {self.detector.hog_orientations}
HOG Pixels per Cell: {self.detector.hog_pixels_per_cell}
HOG Cells per Block: {self.detector.hog_cells_per_block}
LBP Radius: {self.detector.lbp_radius}
LBP Points: {self.detector.lbp_n_points}
Gabor Frequencies: {self.detector.gabor_frequencies}
Gabor Angles: {[f"{np.degrees(a):.0f}°" for a in self.detector.gabor_angles]}

MEMORY OPTIMIZATION SETTINGS
-----------------------------
Max Patches per Image: {self.detector.max_patches_per_image}
Batch Size: {self.detector.batch_size}
Fast Mode: {self.detector.fast_mode}

TRAINING RESULTS
----------------
Training Accuracy: {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall: {results['recall']:.4f}
F1-Score: {results['f1_score']:.4f}

DATASET STATISTICS
------------------
Number of Training Images: {results['n_samples']}
Total Positive Samples: {results['positive_samples']}
Total Negative Samples: {results['negative_samples']}
Feature Vector Size: {results['n_features']}

MODEL PARAMETERS
----------------
Hyperparameter Tuning: {'Yes' if self.tune_hyperparams_var.get() else 'No'}
"""
        
        # ADD THIS NEW SECTION - Shows which hyperparameters were used
        if 'best_params' in results and results['best_params']:
            report_text += "\nOPTIMIZED HYPERPARAMETERS\n"
            report_text += "-------------------------\n"
            for param, value in results['best_params'].items():
                report_text += f"{param}: {value}\n"
            
            if 'best_score' in results and results['best_score']:
                report_text += f"\nBest Cross-Validation F1-Score: {results['best_score']:.4f}\n"
        else:
            # If no hyperparameter tuning, show default parameters
            report_text += "\nDEFAULT MODEL PARAMETERS\n"
            report_text += "------------------------\n"
            report_text += f"n_estimators: {self.detector.rf_params['n_estimators']}\n"
            report_text += f"max_depth: {self.detector.rf_params['max_depth']}\n"
            report_text += f"min_samples_split: {self.detector.rf_params['min_samples_split']}\n"
            report_text += f"min_samples_leaf: {self.detector.rf_params['min_samples_leaf']}\n"
            report_text += f"max_features: {self.detector.rf_params['max_features']}\n"
            report_text += f"random_state: {self.detector.rf_params['random_state']}\n"
        
        # Feature importance analysis
        if 'feature_importance' in results and hasattr(self.detector, 'feature_names'):
            report_text += "\nTOP 10 MOST IMPORTANT FEATURES\n"
            report_text += "------------------------------\n"
            
            importance = results['feature_importance']
            top_indices = np.argsort(importance)[-10:][::-1]
            feature_names = self.detector.feature_names
            
            for rank, idx in enumerate(top_indices, 1):
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    report_text += f"{rank:2d}. {feature_name}: {importance[idx]:.4f}\n"
        
        report_text += f"""
PERFORMANCE INTERPRETATION
---------------------------
- Training Accuracy ({results['accuracy']:.3f}): Model correctly classified {results['accuracy']*100:.1f}% of training samples
- Precision ({results['precision']:.3f}): Of all positive predictions, {results['precision']*100:.1f}% were correct
- Recall ({results['recall']:.3f}): Model found {results['recall']*100:.1f}% of all actual ants
- F1-Score ({results['f1_score']:.3f}): Balanced measure between precision and recall

RECOMMENDATIONS
---------------"""

        if results['accuracy'] < 0.8:
            report_text += "\n- Consider adding more diverse training data"
        if results['precision'] < 0.8:
            report_text += "\n- Model may benefit from more negative samples"
        if results['recall'] < 0.8:
            report_text += "\n- Consider increasing model complexity or adding more positive samples"
        if results['f1_score'] >= 0.8:
            report_text += "\n- Model shows good balanced performance"
        
        report_text += "\n\nNOTES\n-----\n"
        report_text += "This model uses Random Forest classification with comprehensive feature extraction\n"
        report_text += "including HOG, LBP, Gabor filters, and statistical features for robust ant detection.\n"
        
        return report_text

    def save_model(self):
        """Save trained model to file"""
        if self.detector.model is None:
            messagebox.showwarning("Warning", "No trained model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.detector.save_model_to_file(filepath):
                    messagebox.showinfo("Success", f"Model saved to {filepath}")
                    self.update_status(f"Model saved: {os.path.basename(filepath)}")
                else:
                    messagebox.showerror("Error", "Failed to save model")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving model: {str(e)}")

    def load_model(self):
        """Load trained model from file"""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.detector.load_model_from_file(filepath):
                    messagebox.showinfo("Success", f"Model loaded from {filepath}")
                    self.update_status(f"Model loaded: {os.path.basename(filepath)}")
                    
                    # Update model status labels in other tabs
                    if hasattr(self, 'model_status_label'):
                        self.model_status_label.config(text="Model loaded", style='Success.TLabel')
                    if hasattr(self, 'pred_model_status_label'):
                        self.pred_model_status_label.config(text="Model loaded", style='Success.TLabel')
                else:
                    messagebox.showerror("Error", "Failed to load model")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")

 
# Methods to add to AntDetectionApp class (4-space indentation)

    def load_model_for_testing(self):
        """Load trained model from file for testing"""
        filepath = filedialog.askopenfilename(
            title="Load Model for Testing",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.detector.load_model_from_file(filepath):
                    messagebox.showinfo("Success", f"Model loaded from {filepath}")
                    self.update_status(f"Model loaded: {os.path.basename(filepath)}")
                    
                    # Update model status labels
                    self.model_status_label.config(text="Model loaded", style='Success.TLabel')
                    return True
                else:
                    messagebox.showerror("Error", "Failed to load model")
                    return False
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                return False
        return False

    def load_test_images_batch(self):
        """Load multiple test images"""
        filepaths = filedialog.askopenfilenames(
            title="Select Test Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filepaths:
            test_images = []
            failed_images = []
            
            for filepath in filepaths:
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        test_images.append({
                            'image': image,
                            'path': filepath,
                            'filename': os.path.basename(filepath)
                        })
                    else:
                        failed_images.append(filepath)
                except Exception as e:
                    failed_images.append(filepath)
            
            if test_images:
                # Store in instance variable
                if not hasattr(self, 'test_images'):
                    self.test_images = []
                self.test_images.extend(test_images)
                
                self.update_test_info()
                messagebox.showinfo("Success", f"Loaded {len(test_images)} test images")
            
            if failed_images:
                messagebox.showwarning("Warning", f"Failed to load {len(failed_images)} images")

    def load_test_annotations_batch(self):
        """Load multiple test annotation files"""
        filepaths = filedialog.askopenfilenames(
            title="Select Test Annotation Files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepaths:
            test_annotations = []
            failed_annotations = []
            
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Extract image name and annotations
                    image_path = data.get('image_path', os.path.basename(filepath))
                    annotations = data.get('annotations', [])
                    
                    test_annotations.append({
                        'image_path': image_path,
                        'annotations': annotations,
                        'annotation_file': filepath
                    })
                    
                except Exception as e:
                    failed_annotations.append(filepath)
            
            if test_annotations:
                # Store in instance variable
                if not hasattr(self, 'test_annotations'):
                    self.test_annotations = []
                self.test_annotations.extend(test_annotations)
                
                self.update_test_info()
                messagebox.showinfo("Success", f"Loaded {len(test_annotations)} annotation files")
            
            if failed_annotations:
                messagebox.showwarning("Warning", f"Failed to load {len(failed_annotations)} annotation files")

    def update_test_info(self):
        """Update test info display"""
        num_images = len(getattr(self, 'test_images', []))
        num_annotations = len(getattr(self, 'test_annotations', []))
        
        info_text = f"Images: {num_images}, Annotations: {num_annotations}"
        if hasattr(self, 'test_info_label'):
            self.test_info_label.config(text=info_text)

    def view_test_pairs(self):
        """View test image-annotation pairs with pagination"""
        if not hasattr(self, 'test_images') or not hasattr(self, 'test_annotations'):
            messagebox.showwarning("Warning", "Please load test images and annotations first")
            return
        
        # Match images with annotations
        matched_pairs = []
        for img_data in self.test_images:
            img_filename = img_data['filename']
            
            # Find matching annotation
            for ann_data in self.test_annotations:
                ann_image_path = ann_data['image_path']
                if os.path.basename(ann_image_path) == img_filename:
                    matched_pairs.append({
                        'image': img_data,
                        'annotations': ann_data['annotations'],
                        'annotation_file': ann_data['annotation_file']
                    })
                    break
        
        if not matched_pairs:
            messagebox.showinfo("Info", "No matched test pairs found")
            return
        
        # Create new window for viewing pairs
        pairs_window = tk.Toplevel(self.root)
        pairs_window.title(f"Test Pairs - {len(matched_pairs)} total")
        pairs_window.geometry("1400x900")
        
        # Maximize the window
        try:
            pairs_window.state('zoomed')
        except:
            try:
                pairs_window.attributes('-zoomed', True)
            except:
                pass
        
        # Variables for pagination
        pairs_per_page = 10  # Show 10 pairs at a time
        current_page = [0]  # Use list so it's mutable in nested function
        total_pages = (len(matched_pairs) + pairs_per_page - 1) // pairs_per_page
        
        # Main container
        main_container = ttk.Frame(pairs_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Navigation frame at top
        nav_frame = ttk.Frame(main_container)
        nav_frame.pack(fill=tk.X, pady=5)
        
        page_label = ttk.Label(nav_frame, text=f"Page 1 of {total_pages}", style='Heading.TLabel')
        page_label.pack(side=tk.LEFT, padx=10)
        
        # Content frame with scrollbar
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(content_frame)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def show_page(page_num):
            """Display specific page of pairs"""
            # Clear existing content
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Calculate range
            start_idx = page_num * pairs_per_page
            end_idx = min(start_idx + pairs_per_page, len(matched_pairs))
            
            # Update page label
            page_label.config(text=f"Page {page_num + 1} of {total_pages} | Showing pairs {start_idx + 1}-{end_idx} of {len(matched_pairs)}")
            
            # Display pairs for this page
            for i in range(start_idx, end_idx):
                pair = matched_pairs[i]
                
                # Create frame for this pair
                pair_frame = ttk.LabelFrame(scrollable_frame, text=f"Test Pair {i+1}: {pair['image']['filename']}")
                pair_frame.pack(fill=tk.X, padx=5, pady=8)
                
                # Create horizontal layout
                content_frame_inner = ttk.Frame(pair_frame)
                content_frame_inner.pack(fill=tk.X, padx=15, pady=15)
                
                # Left side - Image with annotations
                image_frame = ttk.Frame(content_frame_inner)
                image_frame.pack(side=tk.LEFT, padx=(0, 30))
                
                try:
                    # Get image and draw annotations
                    image = pair['image']['image'].copy()
                    h, w = image.shape[:2]
                    
                    # Draw annotation boxes on image
                    for j, ann in enumerate(pair['annotations']):
                        x1, y1 = int(ann['x']), int(ann['y'])
                        x2, y2 = int(ann['x'] + ann['width']), int(ann['y'] + ann['height'])
                        
                        # Draw rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add label
                        label = f"{j+1}"
                        cv2.putText(image, label, (x1, max(y1-8, 20)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Resize for display
                    display_height = 675
                    aspect_ratio = w / h
                    display_width = int(display_height * aspect_ratio)
                    if display_width > 900:
                        display_width = 900
                        display_height = int(display_width / aspect_ratio)
                    
                    display_image = cv2.resize(image, (display_width, display_height))
                    
                    # Convert to PhotoImage
                    img_pil = Image.fromarray(display_image)
                    photo = ImageTk.PhotoImage(img_pil)
                    
                    # Display image
                    image_label = ttk.Label(image_frame, image=photo)
                    image_label.image = photo
                    image_label.pack()
                    
                    # Image info
                    img_info = ttk.Label(image_frame, text=f"Original Size: {w}x{h} pixels", 
                                       style='Info.TLabel', font=('Arial', 10))
                    img_info.pack(pady=(8, 0))
                    
                except Exception as e:
                    error_label = ttk.Label(image_frame, text=f"Error loading image: {str(e)}", 
                                          style='Error.TLabel')
                    error_label.pack()
                
                # Right side - Annotation details
                info_frame = ttk.Frame(content_frame_inner)
                info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Annotation summary
                summary_label = ttk.Label(info_frame, text=f"Annotations: {len(pair['annotations'])}", 
                                        style='Heading.TLabel', font=('Arial', 14, 'bold'))
                summary_label.pack(anchor=tk.W, pady=(0, 15))
                
                # Create scrollable text for annotation details
                details_frame = ttk.Frame(info_frame)
                details_frame.pack(fill=tk.BOTH, expand=True)
                
                details_text = tk.Text(details_frame, height=18, width=60, wrap=tk.WORD, 
                                     font=('Arial', 10),
                                     bg=self.colors['bg_light'], fg=self.colors['fg'], 
                                     insertbackground=self.colors['accent'],
                                     relief=tk.FLAT, padx=10, pady=10)
                details_scroll = ttk.Scrollbar(details_frame, command=details_text.yview)
                details_text.config(yscrollcommand=details_scroll.set)
                
                details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Add annotation details
                details_text.insert(tk.END, f"Test Annotation Details for {pair['image']['filename']}\n")
                details_text.insert(tk.END, "=" * 50 + "\n\n")
                
                for j, ann in enumerate(pair['annotations']):
                    details_text.insert(tk.END, f"Ant {j+1}:\n")
                    details_text.insert(tk.END, f"  Position: ({ann['x']:.0f}, {ann['y']:.0f})\n")
                    details_text.insert(tk.END, f"  Size: {ann['width']:.0f} x {ann['height']:.0f} pixels\n")
                    details_text.insert(tk.END, f"  Area: {ann['width'] * ann['height']:.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Center: ({ann['x'] + ann['width']/2:.0f}, {ann['y'] + ann['height']/2:.0f})\n")
                    details_text.insert(tk.END, "\n")
                
                # Add summary statistics
                if pair['annotations']:
                    areas = [ann['width'] * ann['height'] for ann in pair['annotations']]
                    details_text.insert(tk.END, "Summary Statistics:\n")
                    details_text.insert(tk.END, f"  Average area: {np.mean(areas):.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Min area: {min(areas):.0f} pxÂ²\n")
                    details_text.insert(tk.END, f"  Max area: {max(areas):.0f} pxÂ²\n")
                
                details_text.config(state=tk.DISABLED)
            
            canvas.yview_moveto(0)  # Scroll to top
        
        def prev_page():
            if current_page[0] > 0:
                current_page[0] -= 1
                show_page(current_page[0])
        
        def next_page():
            if current_page[0] < total_pages - 1:
                current_page[0] += 1
                show_page(current_page[0])
        
        # Navigation buttons
        ttk.Button(nav_frame, text="< Previous Page", command=prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Page >", command=next_page).pack(side=tk.LEFT, padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Show first page
        show_page(0)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)

    def run_evaluation(self):
        """Run comprehensive model evaluation"""
        if self.detector.model is None:
            messagebox.showerror("Error", "No model loaded. Please load a model first.")
            return
        
        if not hasattr(self, 'test_images') or not hasattr(self, 'test_annotations'):
            messagebox.showerror("Error", "Please load test images and annotations first.")
            return
        
        # Match images with annotations
        matched_pairs = []
        for img_data in self.test_images:
            img_filename = img_data['filename']
            
            # Find matching annotation
            for ann_data in self.test_annotations:
                ann_image_path = ann_data['image_path']
                if os.path.basename(ann_image_path) == img_filename:
                    # Convert annotations to required format (pixel coordinates)
                    image = img_data['image']
                    h, w = image.shape[:2]
                    
                    gt_boxes = []
                    for ann in ann_data['annotations']:
                        x1 = float(ann['x'])
                        y1 = float(ann['y'])
                        x2 = float(ann['x'] + ann['width'])
                        y2 = float(ann['y'] + ann['height'])
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        if x2 > x1 and y2 > y1:
                            gt_boxes.append([x1, y1, x2, y2])
                    
                    matched_pairs.append((img_data, gt_boxes))
                    break
        
        if not matched_pairs:
            messagebox.showerror("Error", "No matching image-annotation pairs found")
            return
        
        # Get evaluation parameters
        confidence_threshold = self.test_confidence_var.get()
        distance_threshold = self.test_distance_var.get()
        use_normalized_distance = self.test_normalized_var.get()
        self.detector.fast_mode = self.test_fast_mode_var.get()
        
        # Create evaluation results window
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Evaluation Results")
        eval_window.geometry("1600x1000")
        
        # Maximize window
        try:
            eval_window.state('zoomed')
        except:
            try:
                eval_window.attributes('-zoomed', True)
            except:
                pass
        
        # Create main container
        main_frame = ttk.Frame(eval_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        progress_label = ttk.Label(progress_frame, text="Starting evaluation...")
        progress_label.pack(side=tk.LEFT)
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Update GUI
        eval_window.update()
        
        # Initialize results storage
        all_results = []
        overall_metrics = {
            'total_tp': 0, 'total_fp': 0, 'total_fn': 0,
            'total_predictions': 0, 'total_ground_truth': 0,
            'all_precisions': [], 'all_recalls': [], 'all_f1s': [], 'all_aps': []
        }
        
        # Run evaluation on each image
        try:
            for img_idx, (img_data, gt_boxes) in enumerate(matched_pairs):
                progress = (img_idx + 1) / len(matched_pairs)
                progress_bar['value'] = progress * 100
                progress_label.config(text=f"Evaluating image {img_idx + 1}/{len(matched_pairs)}: {img_data['filename']}")
                eval_window.update()
                
                test_img = img_data['image']
                h, w = test_img.shape[:2]
                
                # Get model predictions
                pred_boxes, pred_scores, _ = self.detector.predict(
                    test_img,
                    confidence_threshold=confidence_threshold,
                    distance_threshold=distance_threshold,
                    use_normalized_distance=use_normalized_distance
                )
                
                # Evaluate predictions
                if use_normalized_distance:
                    img_metrics = self.detector.evaluate_object_detection(
                        pred_boxes, pred_scores, gt_boxes,
                        image_width=w, image_height=h,
                        normalized_threshold=distance_threshold
                    )
                else:
                    img_metrics = self.detector.evaluate_object_detection(
                        pred_boxes, pred_scores, gt_boxes,
                        distance_threshold=distance_threshold
                    )
                
                # Store results
                all_results.append({
                    'image_name': img_data['filename'],
                    'image': test_img,
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'gt_boxes': gt_boxes,
                    'metrics': img_metrics,
                    'image_width': w,
                    'image_height': h
                })
                
                # Accumulate overall metrics
                overall_metrics['total_tp'] += img_metrics['true_positives']
                overall_metrics['total_fp'] += img_metrics['false_positives']
                overall_metrics['total_fn'] += img_metrics['false_negatives']
                overall_metrics['total_predictions'] += img_metrics['num_predictions']
                overall_metrics['total_ground_truth'] += img_metrics['num_ground_truth']
                
                if img_metrics['num_ground_truth'] > 0:
                    overall_metrics['all_precisions'].append(img_metrics['precision'])
                    overall_metrics['all_recalls'].append(img_metrics['recall'])
                    overall_metrics['all_f1s'].append(img_metrics['f1_score'])
                    overall_metrics['all_aps'].append(img_metrics['average_precision'])
            
            # Calculate overall metrics
            overall_precision = overall_metrics['total_tp'] / (overall_metrics['total_tp'] + overall_metrics['total_fp']) if (overall_metrics['total_tp'] + overall_metrics['total_fp']) > 0 else 0
            overall_recall = overall_metrics['total_tp'] / (overall_metrics['total_tp'] + overall_metrics['total_fn']) if (overall_metrics['total_tp'] + overall_metrics['total_fn']) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            mean_ap = np.mean(overall_metrics['all_aps']) if overall_metrics['all_aps'] else 0
            
            # Calculate additional metrics
            all_kappa = [r['metrics']['kappa_score'] for r in all_results]
            all_det_acc = [r['metrics']['detection_accuracy'] for r in all_results]
            mean_kappa = np.mean(all_kappa) if all_kappa else 0.0
            mean_det_accuracy = np.mean(all_det_acc) if all_det_acc else 0.0
            
            # Remove progress bar
            progress_frame.destroy()
            
            # Store results at the APP level for export - ADD THIS
            self.last_evaluation_results = {
                'all_results': all_results,
                'summary_results': {
                    'overall_precision': overall_precision,
                    'overall_recall': overall_recall,
                    'overall_f1': overall_f1,
                    'mean_ap': mean_ap,
                    'mean_kappa': mean_kappa,
                    'mean_det_accuracy': mean_det_accuracy,
                    'overall_metrics': overall_metrics,
                    'config': {
                        'confidence_threshold': confidence_threshold,
                        'distance_threshold': distance_threshold,
                        'use_normalized_distance': use_normalized_distance
                    }
                }
            }
            
            # Create results display
            self.create_evaluation_results_display(eval_window, main_frame, all_results, {
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'mean_ap': mean_ap,
                'mean_kappa': mean_kappa,
                'mean_det_accuracy': mean_det_accuracy,
                'overall_metrics': overall_metrics,
                'config': {
                    'confidence_threshold': confidence_threshold,
                    'distance_threshold': distance_threshold,
                    'use_normalized_distance': use_normalized_distance
                }
            })
            
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Error during evaluation: {str(e)}")
            eval_window.destroy()

    def setup_distance_threshold_ui(self):
        """Setup the TEST distance threshold UI elements (for testing tab)"""
        # Clear existing widgets
        for widget in self.distance_threshold_frame.winfo_children():
            widget.destroy()
        
        use_normalized = self.test_normalized_var.get()
        
        if use_normalized:
            # Normalized distance threshold (0.005 to 0.20)
            ttk.Label(self.distance_threshold_frame, text="Normalized Distance:").grid(row=0, column=0, sticky=tk.W)
            
            # Create frame for slider and labels
            slider_frame = ttk.Frame(self.distance_threshold_frame)
            slider_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            slider_frame.columnconfigure(1, weight=1)
            
            # Min label
            ttk.Label(slider_frame, text="0.005", style='Info.TLabel').grid(row=0, column=0)
            
            # Slider - FIXED: uses test_distance_var
            distance_scale = tk.Scale(slider_frame, from_=0.005, to=0.20, 
                                      variable=self.test_distance_var, orient=tk.HORIZONTAL, resolution=0.005,
                                      bg=self.colors['bg'], fg=self.colors['fg'], 
                                      troughcolor=self.colors['bg_light'],
                                      activebackground=self.colors['accent'],
                                      highlightthickness=0, sliderrelief=tk.FLAT)
            distance_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            
            # Max label
            ttk.Label(slider_frame, text="0.20", style='Info.TLabel').grid(row=0, column=2)
            
            # Current value label
            self.test_distance_value_label = ttk.Label(self.distance_threshold_frame, text="0.050", style='Info.TLabel')
            self.test_distance_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
            
            # Help text
            help_text = "5% = 5% of image diagonal (recommended)"
            ttk.Label(self.distance_threshold_frame, text=help_text, style='Info.TLabel').grid(row=1, column=1, sticky=tk.W, padx=5)
            
            # Update function for normalized
            def update_normalized_distance_label(*args):
                value = self.test_distance_var.get()
                self.test_distance_value_label.config(text=f"{value:.3f}")
            
            self.test_distance_var.trace('w', update_normalized_distance_label)
            update_normalized_distance_label()  # Initialize
            
        else:
            # Pixel distance threshold (10 to 200)
            ttk.Label(self.distance_threshold_frame, text="Pixel Distance:").grid(row=0, column=0, sticky=tk.W)
            
            # Create frame for slider and labels
            slider_frame = ttk.Frame(self.distance_threshold_frame)
            slider_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            slider_frame.columnconfigure(1, weight=1)
            
            # Min label
            ttk.Label(slider_frame, text="10", style='Info.TLabel').grid(row=0, column=0)
            
            # Slider - FIXED: uses test_distance_var
            distance_scale = tk.Scale(slider_frame, from_=10, to=200, 
                                      variable=self.test_distance_var, orient=tk.HORIZONTAL, resolution=1,
                                      bg=self.colors['bg'], fg=self.colors['fg'], 
                                      troughcolor=self.colors['bg_light'],
                                      activebackground=self.colors['accent'],
                                      highlightthickness=0, sliderrelief=tk.FLAT)
            distance_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            
            # Max label
            ttk.Label(slider_frame, text="200", style='Info.TLabel').grid(row=0, column=2)
            
            # Current value label
            self.test_distance_value_label = ttk.Label(self.distance_threshold_frame, text="50", style='Info.TLabel')
            self.test_distance_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
            
            # Help text
            help_text = "Fixed pixel distance (good for same-size images)"
            ttk.Label(self.distance_threshold_frame, text=help_text, style='Info.TLabel').grid(row=1, column=1, sticky=tk.W, padx=5)
            
            # Update function for pixels
            def update_pixel_distance_label(*args):
                value = self.test_distance_var.get()
                self.test_distance_value_label.config(text=f"{int(value)}")
            
            self.test_distance_var.trace('w', update_pixel_distance_label)
            update_pixel_distance_label()  # Initialize

    def update_distance_threshold_ui(self):
        """Update distance threshold UI when normalized checkbox changes"""
        use_normalized = self.test_normalized_var.get()
        
        if use_normalized:
            # Switch to normalized mode - set reasonable default
            self.test_distance_var.set(0.05)
        else:
            # Switch to pixel mode - set reasonable default
            self.test_distance_var.set(50)
        
        # Rebuild the UI
        self.setup_distance_threshold_ui()

    def export_test_results(self):
        """Export test results - wrapper for the main export function"""
        if hasattr(self, 'last_evaluation_results'):
            self.export_evaluation_results(
                self.last_evaluation_results['all_results'],
                self.last_evaluation_results['summary_results']
            )
        else:
            messagebox.showwarning("Warning", "No evaluation results to export. Please run evaluation first.")

    def create_testing_tab(self):
        """Create testing tab with comprehensive testing controls matching Streamlit version"""
        testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(testing_frame, text="Testing")
        
        # Configure grid
        testing_frame.columnconfigure(1, weight=1)
        testing_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(testing_frame, text="Testing Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        current_row = 0
        
        # Model loading section
        ttk.Label(control_frame, text="Model:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        ttk.Button(control_frame, text="Load Trained Model", command=self.load_model_for_testing).grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        current_row += 1
        
        self.model_status_label = ttk.Label(control_frame, text="No model loaded", style='Info.TLabel')
        self.model_status_label.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Test data loading section
        ttk.Separator(control_frame, orient='horizontal').grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        ttk.Label(control_frame, text="Test Data:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        ttk.Button(control_frame, text="Load Test Images (Batch)", 
                  command=self.load_test_images_batch).grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        current_row += 1
        
        ttk.Button(control_frame, text="Load Test Annotations (Batch)", 
                  command=self.load_test_annotations_batch).grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        current_row += 1
        
        self.test_info_label = ttk.Label(control_frame, text="No test data loaded", style='Info.TLabel')
        self.test_info_label.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        ttk.Button(control_frame, text="View Test Pairs", command=self.view_test_pairs).grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        current_row += 1
        
        # Test Configuration section
        ttk.Separator(control_frame, orient='horizontal').grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        ttk.Label(control_frame, text="Test Configuration:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        # Confidence Threshold with value display
        conf_frame = ttk.Frame(control_frame)
        conf_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        conf_frame.columnconfigure(1, weight=1)
        
        ttk.Label(conf_frame, text="Detection Confidence:").grid(row=0, column=0, sticky=tk.W)
        
        self.test_confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, variable=self.test_confidence_var, 
                                    orient=tk.HORIZONTAL, resolution=0.01,
                                    bg=self.colors['bg'], fg=self.colors['fg'], 
                                    troughcolor=self.colors['bg_light'],
                                    activebackground=self.colors['accent'],
                                    highlightthickness=0, sliderrelief=tk.FLAT)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.test_confidence_value_label = ttk.Label(conf_frame, text="0.600", style='Info.TLabel')
        self.test_confidence_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Update label when slider changes
        def update_confidence_label(*args):
            value = self.test_confidence_var.get()
            self.test_confidence_value_label.config(text=f"{value:.3f}")
        
        self.test_confidence_var.trace('w', update_confidence_label)
        current_row += 1
        
        # Distance evaluation method selection
        ttk.Label(control_frame, text="Distance Evaluation Method:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        current_row += 1
        
        # Use normalized distance checkbox
        self.test_normalized_var = tk.BooleanVar(value=True)
        normalized_check = ttk.Checkbutton(control_frame, text="Use Normalized Distance Threshold", 
                                          variable=self.test_normalized_var,
                                          command=self.update_distance_threshold_ui)
        normalized_check.grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Distance threshold frame (will be updated based on normalized choice)
        self.distance_threshold_frame = ttk.Frame(control_frame)
        self.distance_threshold_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.distance_threshold_frame.columnconfigure(1, weight=1)
        current_row += 1
        
        # Initialize distance threshold UI
        self.test_distance_var = tk.DoubleVar(value=0.05)
        self.setup_distance_threshold_ui()
        
        # Compact help text for distance thresholds
        help_text = "Normalized (5%): % of diagonal | Pixel: Fixed pixels"
        ttk.Label(control_frame, text=help_text, style='Info.TLabel', font=('Arial', 8)).grid(
            row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Additional test parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        ttk.Label(control_frame, text="Additional Parameters:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        # Fast detection mode
        self.test_fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Fast Detection Mode", 
                       variable=self.test_fast_mode_var).grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Show detailed visualization option
        self.show_detailed_viz_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Show Detailed Visualizations", 
                       variable=self.show_detailed_viz_var).grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=2)
        current_row += 1
        
        # Evaluation controls
        ttk.Separator(control_frame, orient='horizontal').grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        ttk.Label(control_frame, text="Evaluation:", style='Heading.TLabel').grid(row=current_row, column=0, columnspan=2, sticky=tk.W, pady=5)
        current_row += 1
        
        # Place both buttons on same row
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Run Evaluation", command=self.run_evaluation).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))
        ttk.Button(button_frame, text="Export Results", command=self.export_test_results).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 0))
        current_row += 1
        
        # IMPORTANT: Create an empty row AFTER all content, then apply weight to it
        # This ensures the expandable space comes after all buttons and content
        current_row += 1  # Move to empty row
        control_frame.rowconfigure(current_row, weight=1)  # Apply weight to empty row
        
        # Right panel - Results (keep original results panel)
        self.create_test_results_panel(testing_frame, 0, 1)

    def create_evaluation_results_display(self, window, parent, all_results, summary_results):
        """Create comprehensive evaluation results display"""
        # Create notebook for different result views
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text="Summary")
        
        # Overall metrics display
        metrics_frame = ttk.LabelFrame(summary_frame, text="Overall Performance Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create metrics grid
        metrics_data = [
            ("Overall Precision", f"{summary_results['overall_precision']:.3f}"),
            ("Overall Recall", f"{summary_results['overall_recall']:.3f}"),
            ("Overall F1-Score", f"{summary_results['overall_f1']:.3f}"),
            ("Mean Average Precision", f"{summary_results['mean_ap']:.3f}"),
            ("Mean Kappa Score", f"{summary_results['mean_kappa']:.3f}"),
            ("Mean Detection Accuracy", f"{summary_results['mean_det_accuracy']:.3f}"),
            ("Total True Positives", str(summary_results['overall_metrics']['total_tp'])),
            ("Total False Positives", str(summary_results['overall_metrics']['total_fp'])),
            ("Total False Negatives", str(summary_results['overall_metrics']['total_fn'])),
            ("Total Predictions", str(summary_results['overall_metrics']['total_predictions'])),
            ("Total Ground Truth", str(summary_results['overall_metrics']['total_ground_truth'])),
            ("Images Evaluated", str(len(all_results)))
        ]
        
        for i, (metric, value) in enumerate(metrics_data):
            row = i // 3
            col = (i % 3) * 2
            
            ttk.Label(metrics_frame, text=f"{metric}:", style='Info.TLabel').grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2)
            ttk.Label(metrics_frame, text=value, style='Info.TLabel', font=('Arial', 9, 'bold')).grid(
                row=row, column=col+1, sticky=tk.W, padx=10, pady=2)
        
        # Configuration display
        config_frame = ttk.LabelFrame(summary_frame, text="Evaluation Configuration", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        config_text = f"""
Confidence Threshold: {summary_results['config']['confidence_threshold']:.3f}
Distance Threshold: {summary_results['config']['distance_threshold']:.3f}
Distance Method: {'Normalized' if summary_results['config']['use_normalized_distance'] else 'Pixel-based'}
"""
        
        ttk.Label(config_frame, text=config_text.strip(), style='Info.TLabel', justify=tk.LEFT).pack(anchor=tk.W)
        
        # Per-image results tab
        per_image_frame = ttk.Frame(results_notebook)
        results_notebook.add(per_image_frame, text="Per-Image Results")
        
        # Create treeview for per-image results
        tree_frame = ttk.Frame(per_image_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview with scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        columns = ("Image", "GT", "Pred", "TP", "FP", "FN", "Precision", "Recall", "F1", "AP")
        results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                   yscrollcommand=tree_scroll_y.set,
                                   xscrollcommand=tree_scroll_x.set)
        
        # Configure scrollbars
        tree_scroll_y.config(command=results_tree.yview)
        tree_scroll_x.config(command=results_tree.xview)
        
        # Style the treeview - FIX IS HERE
        tree_style = ttk.Style()
        tree_style.configure("Treeview",
                            background=self.colors['bg_light'],
                            foreground=self.colors['fg'],
                            fieldbackground=self.colors['bg_light'],
                            font=('Arial', 9))
        tree_style.configure("Treeview.Heading",
                            background=self.colors['accent'],
                            foreground=self.colors['button_text'],
                            font=('Arial', 9, 'bold'))
        tree_style.map('Treeview',
                      background=[('selected', self.colors['secondary'])],
                      foreground=[('selected', self.colors['button_text'])])
        
        # Configure column headings and widths
        column_widths = {"Image": 150, "GT": 40, "Pred": 40, "TP": 40, "FP": 40, "FN": 40,
                        "Precision": 80, "Recall": 80, "F1": 80, "AP": 80}
        
        for col in columns:
            results_tree.heading(col, text=col)
            results_tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Populate treeview
        for result in all_results:
            metrics = result['metrics']
            values = (
                result['image_name'],
                str(metrics['num_ground_truth']),
                str(metrics['num_predictions']),
                str(metrics['true_positives']),
                str(metrics['false_positives']),
                str(metrics['false_negatives']),
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['average_precision']:.3f}"
            )
            results_tree.insert("", tk.END, values=values)
        
        results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        viz_frame = ttk.Frame(results_notebook)
        results_notebook.add(viz_frame, text="Visualizations")
        
        # Add controls for image navigation
        nav_frame = ttk.Frame(viz_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.viz_index = 0
        
        ttk.Button(nav_frame, text="< Previous", command=lambda: self.show_viz_image(-1)).pack(side=tk.LEFT)
        
        self.viz_label = ttk.Label(nav_frame, text=f"Image 1 of {len(all_results)}", style='Info.TLabel')
        self.viz_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(nav_frame, text="Next >", command=lambda: self.show_viz_image(1)).pack(side=tk.LEFT)
        
        # Canvas for visualization
        self.viz_canvas = tk.Canvas(viz_frame, bg='white', height=500,
                                   highlightthickness=2, highlightbackground=self.colors['accent'])
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Store results for visualization
        self.viz_results = all_results
        
        # Show first image
        if all_results:
            self.show_visualization_image(0)

    def show_viz_image(self, direction):
        """Navigate through visualization images"""
        if not hasattr(self, 'viz_results') or not self.viz_results:
            return
        
        self.viz_index = max(0, min(len(self.viz_results) - 1, self.viz_index + direction))
        self.viz_label.config(text=f"Image {self.viz_index + 1} of {len(self.viz_results)}")
        self.show_visualization_image(self.viz_index)

    def show_visualization_image(self, index):
        """Show visualization for specific image"""
        if not hasattr(self, 'viz_results') or index >= len(self.viz_results):
            return
        
        result = self.viz_results[index]
        
        try:
            # Use the detector's visualization method
            fig = self.detector.visualize_test_results(
                result['image'],
                result['pred_boxes'],
                result['pred_scores'],
                result['gt_boxes'],
                distance_threshold=result['metrics']['distance_threshold'],
                use_normalized=result['metrics']['threshold_type'] == 'normalized',
                image_width=result['image_width'],
                image_height=result['image_height']
            )
            
            # Convert matplotlib figure to tkinter-compatible image
            import io
            from PIL import Image, ImageTk
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            img = Image.open(buf)
            photo = ImageTk.PhotoImage(img)
            
            # Clear canvas and display image
            self.viz_canvas.delete("all")
            self.viz_canvas.create_image(10, 10, anchor=tk.NW, image=photo)
            self.viz_canvas.image = photo  # Keep reference
            
            # Update canvas scroll region
            self.viz_canvas.config(scrollregion=self.viz_canvas.bbox("all"))
            
            plt.close(fig)  # Clean up matplotlib figure
            
        except Exception as e:
            # Fallback: show error message
            self.viz_canvas.delete("all")
            self.viz_canvas.create_text(200, 100, text=f"Error displaying image: {str(e)}", 
                                       fill="red", font=("Arial", 12))

    def export_evaluation_results(self, all_results, summary_results):
        """Export evaluation results to CSV and detailed report"""
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ask user for save location
        base_filepath = filedialog.asksaveasfilename(
            title="Save Evaluation Results",
            defaultextension=".csv",
            initialfile=f"ant_detection_evaluation_{timestamp}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not base_filepath:
            return
        
        try:
            # Prepare CSV data
            csv_data = []
            headers = [
                "Image_Name", "Ground_Truth_Count", "Predictions_Count",
                "True_Positives", "False_Positives", "False_Negatives",
                "Precision", "Recall", "F1_Score", "Average_Precision",
                "Kappa_Score", "Detection_Accuracy", "Mean_Match_Distance",
                "Image_Width", "Image_Height"
            ]
            
            csv_data.append(headers)
            
            for result in all_results:
                metrics = result['metrics']
                row = [
                    result['image_name'],
                    metrics['num_ground_truth'],
                    metrics['num_predictions'],
                    metrics['true_positives'],
                    metrics['false_positives'],
                    metrics['false_negatives'],
                    round(metrics['precision'], 4),
                    round(metrics['recall'], 4),
                    round(metrics['f1_score'], 4),
                    round(metrics['average_precision'], 4),
                    round(metrics['kappa_score'], 4),
                    round(metrics['detection_accuracy'], 4),
                    round(metrics['mean_match_distance'], 4),
                    result['image_width'],
                    result['image_height']
                ]
                csv_data.append(row)
            
            # Save CSV file
            import csv
            with open(base_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            # Create detailed report
            report_filepath = base_filepath.replace('.csv', '_detailed_report.txt')
            report_content = self.create_evaluation_report(all_results, summary_results)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            messagebox.showinfo("Export Successful", 
                               f"Results exported successfully:\n\n"
                               f"CSV: {base_filepath}\n"
                               f"Report: {report_filepath}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")

    def create_evaluation_report(self, all_results, summary_results):
        """Create detailed evaluation report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""Random Forest Ant Detection Model Evaluation Report
=======================================================
Generated: {timestamp}

EVALUATION SUMMARY
------------------
Images Evaluated: {len(all_results)}
Overall Precision: {summary_results['overall_precision']:.4f}
Overall Recall: {summary_results['overall_recall']:.4f}
Overall F1-Score: {summary_results['overall_f1']:.4f}
Mean Average Precision (mAP): {summary_results['mean_ap']:.4f}
Mean Kappa Score: {summary_results['mean_kappa']:.4f}
Mean Detection Accuracy: {summary_results['mean_det_accuracy']:.4f}

EVALUATION CONFIGURATION
------------------------
Confidence Threshold: {summary_results['config']['confidence_threshold']:.3f}
Distance Threshold: {summary_results['config']['distance_threshold']:.3f}
Distance Method: {'Normalized' if summary_results['config']['use_normalized_distance'] else 'Pixel-based'}

AGGREGATE STATISTICS
--------------------
Total True Positives: {summary_results['overall_metrics']['total_tp']}
Total False Positives: {summary_results['overall_metrics']['total_fp']}
Total False Negatives: {summary_results['overall_metrics']['total_fn']}
Total Predictions Made: {summary_results['overall_metrics']['total_predictions']}
Total Ground Truth Objects: {summary_results['overall_metrics']['total_ground_truth']}

PER-IMAGE RESULTS
-----------------
"""
        
        for i, result in enumerate(all_results, 1):
            metrics = result['metrics']
            report += f"""
Image {i}: {result['image_name']}
  Dimensions: {result['image_width']}x{result['image_height']}
  Ground Truth: {metrics['num_ground_truth']} | Predictions: {metrics['num_predictions']}
  TP: {metrics['true_positives']} | FP: {metrics['false_positives']} | FN: {metrics['false_negatives']}
  Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}
  Average Precision: {metrics['average_precision']:.3f}
  Mean Match Distance: {metrics['mean_match_distance']:.3f}
"""
        
        # Performance analysis
        precisions = [r['metrics']['precision'] for r in all_results if r['metrics']['num_ground_truth'] > 0]
        recalls = [r['metrics']['recall'] for r in all_results if r['metrics']['num_ground_truth'] > 0]
        f1s = [r['metrics']['f1_score'] for r in all_results if r['metrics']['num_ground_truth'] > 0]
        
        if precisions and recalls and f1s:
            report += f"""
PERFORMANCE ANALYSIS
--------------------
Precision Statistics:
  Mean: {np.mean(precisions):.3f} | Std: {np.std(precisions):.3f}
  Min: {np.min(precisions):.3f} | Max: {np.max(precisions):.3f}

Recall Statistics:
  Mean: {np.mean(recalls):.3f} | Std: {np.std(recalls):.3f}
  Min: {np.min(recalls):.3f} | Max: {np.max(recalls):.3f}

F1-Score Statistics:
  Mean: {np.mean(f1s):.3f} | Std: {np.std(f1s):.3f}
  Min: {np.min(f1s):.3f} | Max: {np.max(f1s):.3f}
"""
        
        report += """
INTERPRETATION GUIDE
--------------------
- Precision: Of all detections, what percentage were correct?
- Recall: Of all actual ants, what percentage were found?
- F1-Score: Balanced measure between precision and recall
- Average Precision (AP): Area under precision-recall curve
- Kappa Score: Agreement measure accounting for chance
"""
        
        return report

    def setup_pred_distance_ui(self):
        """Setup the prediction distance threshold UI elements"""
        # Clear existing widgets
        for widget in self.pred_distance_frame.winfo_children():
            widget.destroy()
        
        use_normalized = self.pred_normalized_var.get()
        
        if use_normalized:
            # Normalized distance threshold (0.01 to 0.20)
            ttk.Label(self.pred_distance_frame, text="Normalized Distance:").grid(row=0, column=0, sticky=tk.W)
            
            # Create frame for slider and labels
            slider_frame = ttk.Frame(self.pred_distance_frame)
            slider_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            slider_frame.columnconfigure(1, weight=1)
            
            # Min label
            ttk.Label(slider_frame, text="0.005", style='Info.TLabel').grid(row=0, column=0)
            
            # Slider
            distance_scale = tk.Scale(slider_frame, from_=0.005, to=0.20, 
                                      variable=self.pred_distance_var, orient=tk.HORIZONTAL, resolution=0.005,
                                      bg=self.colors['bg'], fg=self.colors['fg'], 
                                      troughcolor=self.colors['bg_light'],
                                      activebackground=self.colors['accent'],
                                      highlightthickness=0, sliderrelief=tk.FLAT)
            distance_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            
            # Max label
            ttk.Label(slider_frame, text="0.20", style='Info.TLabel').grid(row=0, column=2)
            
            # Current value label
            self.pred_distance_value_label = ttk.Label(self.pred_distance_frame, text="0.050", style='Info.TLabel')
            self.pred_distance_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
            
            # Help text
            help_text = "5% = 5% of image diagonal (recommended)"
            ttk.Label(self.pred_distance_frame, text=help_text, style='Info.TLabel').grid(
                row=1, column=1, sticky=tk.W, padx=5)
            
            # Update function for normalized
            def update_pred_normalized_distance_label(*args):
                value = self.pred_distance_var.get()
                self.pred_distance_value_label.config(text=f"{value:.3f}")
            
            self.pred_distance_var.trace('w', update_pred_normalized_distance_label)
            update_pred_normalized_distance_label()  # Initialize
            
        else:
            # Pixel distance threshold (10 to 200)
            ttk.Label(self.pred_distance_frame, text="Pixel Distance:").grid(row=0, column=0, sticky=tk.W)
            
            # Create frame for slider and labels
            slider_frame = ttk.Frame(self.pred_distance_frame)
            slider_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            slider_frame.columnconfigure(1, weight=1)
            
            # Min label
            ttk.Label(slider_frame, text="10", style='Info.TLabel').grid(row=0, column=0)
            
            # Slider
            distance_scale = tk.Scale(slider_frame, from_=10, to=200, 
                                      variable=self.pred_distance_var, orient=tk.HORIZONTAL, resolution=0.005,
                                      bg=self.colors['bg'], fg=self.colors['fg'], 
                                      troughcolor=self.colors['bg_light'],
                                      activebackground=self.colors['accent'],
                                      highlightthickness=0, sliderrelief=tk.FLAT)
            distance_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
            
            # Max label
            ttk.Label(slider_frame, text="200", style='Info.TLabel').grid(row=0, column=2)
            
            # Current value label
            self.pred_distance_value_label = ttk.Label(self.pred_distance_frame, text="50", style='Info.TLabel')
            self.pred_distance_value_label.grid(row=0, column=2, sticky=tk.W, padx=5)
            
            # Help text
            help_text = "Fixed pixel distance (good for same-size images)"
            ttk.Label(self.pred_distance_frame, text=help_text, style='Info.TLabel').grid(
                row=1, column=1, sticky=tk.W, padx=5)
            
            # Update function for pixels
            def update_pred_pixel_distance_label(*args):
                value = self.pred_distance_var.get()
                self.pred_distance_value_label.config(text=f"{int(value)}")
            
            self.pred_distance_var.trace('w', update_pred_pixel_distance_label)
            update_pred_pixel_distance_label()  # Initialize

    def update_pred_distance_ui(self):
        """Update prediction distance threshold UI when normalized checkbox changes"""
        use_normalized = self.pred_normalized_var.get()
        
        if use_normalized:
            # Switch to normalized mode - set reasonable default
            self.pred_distance_var.set(0.05)
        else:
            # Switch to pixel mode - set reasonable default
            self.pred_distance_var.set(50)
        
        # Rebuild the UI
        self.setup_pred_distance_ui()

    def choose_detection_color(self):
        """Open color chooser for detection boxes"""
        color = colorchooser.askcolor(
            title="Choose Detection Box Color",
            initialcolor=self.detection_color_var.get()
        )
        if color[1]:  # color[1] is the hex value
            self.detection_color_var.set(color[1])

    def load_prediction_images_batch(self):
        """Load multiple images for prediction"""
        filepaths = filedialog.askopenfilenames(
            title="Select Images for Prediction",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filepaths:
            prediction_images = []
            failed_images = []
            
            for filepath in filepaths:
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        prediction_images.append({
                            'image': image,
                            'path': filepath,
                            'filename': os.path.basename(filepath)
                        })
                    else:
                        failed_images.append(filepath)
                except Exception as e:
                    failed_images.append(filepath)
            
            if prediction_images:
                # Store in instance variable
                self.prediction_images = prediction_images
                self.pred_index = 0
                self.prediction_results = None  # Clear previous results
                
                self.update_prediction_info()
                messagebox.showinfo("Success", f"Loaded {len(prediction_images)} images for prediction")
            
            if failed_images:
                messagebox.showwarning("Warning", f"Failed to load {len(failed_images)} images")

    def update_prediction_info(self):
        """Update prediction info display"""
        if hasattr(self, 'prediction_images'):
            num_images = len(self.prediction_images)
            info_text = f"Loaded {num_images} images for prediction"
            self.pred_info_label.config(text=info_text)
            self.pred_index_label.config(text=f"{self.pred_index + 1}/{num_images}")
        else:
            self.pred_info_label.config(text="No images loaded")
            self.pred_index_label.config(text="0/0")

    def run_batch_detection(self):
        """Run detection on all loaded images - full implementation from Streamlit"""
        if self.detector.model is None:
            messagebox.showerror("Error", "No model loaded. Please load a model first.")
            return
        
        if not hasattr(self, 'prediction_images') or not self.prediction_images:
            messagebox.showerror("Error", "No images loaded. Please load images first.")
            return
        
        # Get parameters from UI
        confidence_threshold = self.pred_confidence_var.get()
        distance_threshold = self.pred_distance_var.get()
        use_normalized_distance = self.pred_normalized_var.get()
        fast_mode = self.pred_fast_mode_var.get()
        show_all_detections = self.show_all_detections_var.get()
        
        # Set detector fast mode
        self.detector.fast_mode = fast_mode
        
        # Get detection color (convert hex to RGB)
        color_hex = self.detection_color_var.get()
        detection_color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Detections")
        progress_window.geometry("500x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 100,
            self.root.winfo_rooty() + 100
        ))
        
        # Progress window contents
        ttk.Label(progress_window, text="Processing batch detection...", 
                 style='Heading.TLabel').pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                     maximum=100, length=400)
        progress_bar.pack(pady=10)
        
        status_label = ttk.Label(progress_window, text="Starting...", style='Info.TLabel')
        status_label.pack(pady=5)
        
        cancel_button = ttk.Button(progress_window, text="Cancel", 
                                  command=lambda: setattr(self, '_cancel_detection', True))
        cancel_button.pack(pady=10)
        
        # Initialize cancellation flag
        self._cancel_detection = False
        
        # Update progress window
        progress_window.update()
        
        # Initialize results storage
        all_results = []
        batch_summary = {
            "total_images": len(self.prediction_images), 
            "total_ants": 0, 
            "processed_images": 0,
            "processing_time": 0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Process each image
            for img_idx, img_data in enumerate(self.prediction_images):
                if self._cancel_detection:
                    break
                
                # Update progress
                progress = (img_idx / len(self.prediction_images)) * 100
                progress_var.set(progress)
                status_label.config(text=f"Processing {img_idx + 1}/{len(self.prediction_images)}: {img_data['filename']}")
                progress_window.update()
                
                try:
                    # Run detection on this image
                    image = img_data['image']
                    
                    # Get predictions
                    boxes, scores, all_detections = self.detector.predict(
                        image,
                        confidence_threshold=confidence_threshold,
                        distance_threshold=distance_threshold,
                        use_normalized_distance=use_normalized_distance
                    )
                    
                    # Store results for this image
                    image_results = {
                        "image_name": img_data['filename'],
                        "image": image,
                        "boxes": boxes,
                        "scores": scores,
                        "all_detections": all_detections if show_all_detections else [],
                        "ant_count": len(boxes),
                        "detection_color_rgb": detection_color_rgb
                    }
                    all_results.append(image_results)
                    
                    # Update batch summary
                    batch_summary["total_ants"] += len(boxes)
                    batch_summary["processed_images"] += 1
                    
                except Exception as e:
                    print(f"Error processing {img_data['filename']}: {str(e)}")
                    # Add empty result for failed image
                    all_results.append({
                        "image_name": img_data['filename'],
                        "image": img_data['image'],
                        "boxes": [],
                        "scores": [],
                        "all_detections": [],
                        "ant_count": 0,
                        "error": str(e),
                        "detection_color_rgb": detection_color_rgb
                    })
            
            # Calculate processing time
            batch_summary["processing_time"] = time.time() - start_time
            
            # Store results
            self.prediction_results = {
                "all_results": all_results,
                "batch_summary": batch_summary,
                "parameters": {
                    "confidence_threshold": confidence_threshold,
                    "distance_threshold": distance_threshold,
                    "use_normalized_distance": use_normalized_distance,
                    "fast_mode": fast_mode,
                    "show_all_detections": show_all_detections,
                    "detection_color_rgb": detection_color_rgb
                }
            }
            
            # Close progress window
            progress_window.destroy()
            
            # Update UI with results
            self.update_prediction_results_display()
            self.export_summary_btn.config(state='normal')
            self.export_detailed_btn.config(state='normal')
            
            # Show completion message
            if not self._cancel_detection:
                avg_ants = batch_summary["total_ants"] / max(batch_summary["processed_images"], 1)
                messagebox.showinfo("Detection Complete", 
                                  f"Processed {batch_summary['processed_images']} images\n"
                                  f"Found {batch_summary['total_ants']} total ants\n"
                                  f"Average: {avg_ants:.1f} ants per image\n"
                                  f"Processing time: {batch_summary['processing_time']:.1f} seconds")
            else:
                messagebox.showinfo("Detection Cancelled", "Processing was cancelled by user.")
        
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Detection Error", f"Error during batch detection: {str(e)}")
        
        # Clean up
        if hasattr(self, '_cancel_detection'):
            delattr(self, '_cancel_detection')

    def update_prediction_results_display(self):
        """Update the prediction results display"""
        if not hasattr(self, 'prediction_results') or not self.prediction_results:
            return
        
        batch_summary = self.prediction_results["batch_summary"]
        
        # Update summary metrics
        self.pred_summary_metrics['images_processed'].config(text=str(batch_summary["processed_images"]))
        self.pred_summary_metrics['total_ants_found'].config(text=str(batch_summary["total_ants"]))
        
        avg_ants = batch_summary["total_ants"] / max(batch_summary["processed_images"], 1)
        self.pred_summary_metrics['average_per_image'].config(text=f"{avg_ants:.1f}")
        self.pred_summary_metrics['processing_time'].config(text=f"{batch_summary['processing_time']:.1f}s")
        
        # Show first image if available
        if self.prediction_results["all_results"]:
            self.show_prediction_image(0)

    def show_prediction_image(self, index):
        """Show prediction result for specific image"""
        if (not hasattr(self, 'prediction_results') or 
            not self.prediction_results or 
            index >= len(self.prediction_results["all_results"])):
            return
        
        result = self.prediction_results["all_results"][index]
        self.pred_index = index
        
        try:
            # Update image info
            self.current_image_label.config(text=f"Current: {result['image_name']}")
            self.detection_count_label.config(text=f"Detections: {result['ant_count']}")
            self.pred_index_label.config(text=f"{index + 1}/{len(self.prediction_results['all_results'])}")
            
            # Create visualization at full resolution
            if result["ant_count"] > 0:
                # Draw detections on image
                vis_image_full = self.visualize_predictions(
                    result["image"],
                    result["boxes"],
                    result["scores"],
                    color=result.get("detection_color_rgb", (255, 0, 0))
                )
            else:
                vis_image_full = result["image"].copy()
            
            # Store full resolution image for download
            self.current_pred_image_full_res = Image.fromarray(vis_image_full)
            
            # Convert to PhotoImage for display
            # Resize if too large for display
            h, w = vis_image_full.shape[:2]
            max_display_height = 400
            if h > max_display_height:
                aspect_ratio = w / h
                display_width = int(max_display_height * aspect_ratio)
                vis_image_display = cv2.resize(vis_image_full, (display_width, max_display_height))
            else:
                vis_image_display = vis_image_full
            
            img_pil = Image.fromarray(vis_image_display)
            self.current_pred_image_pil = img_pil
            photo = ImageTk.PhotoImage(img_pil)
            
            # Clear canvas and display
            self.pred_canvas.delete("all")
            self.pred_canvas.create_image(10, 10, anchor=tk.NW, image=photo)
            self.pred_canvas.image = photo  # Keep reference
            
        except Exception as e:
            # Fallback: show error message
            self.pred_canvas.delete("all")
            self.pred_canvas.create_text(200, 100, text=f"Error displaying image: {str(e)}", 
                                       fill="red", font=("Arial", 12))

    def visualize_predictions(self, image, boxes, scores, color=(255, 0, 0), thickness=2):
        """Visualize predictions on image - from Streamlit version"""
        vis_image = image.copy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
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

    def prev_prediction(self):
        """Navigate to previous prediction image"""
        if (hasattr(self, 'prediction_results') and 
            self.prediction_results and 
            self.pred_index > 0):
            self.show_prediction_image(self.pred_index - 1)

    def next_prediction(self):
        """Navigate to next prediction image"""
        if (hasattr(self, 'prediction_results') and 
            self.prediction_results and 
            self.pred_index < len(self.prediction_results["all_results"]) - 1):
            self.show_prediction_image(self.pred_index + 1)

    def export_detection_results(self):
        """Export detection results to CSV - from Streamlit version"""
        if not hasattr(self, 'prediction_results') or not self.prediction_results:
            messagebox.showwarning("Warning", "No prediction results to export. Please run detection first.")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ant_detection_results_{timestamp}.csv"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            title="Save Detection Results",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Prepare data for CSV export
                all_results = self.prediction_results["all_results"]
                
                # Create comprehensive results table
                table_data = []
                detection_id = 1
                
                for result in all_results:
                    image_name = result["image_name"]
                    boxes = result["boxes"]
                    scores = result["scores"]
                    
                    if len(boxes) == 0:
                        # Add row even for images with no detections
                        table_data.append({
                            "Detection_ID": f"IMG_{len(table_data)+1:03d}",
                            "Image_Name": image_name,
                            "Ant_Count": 0,
                            "Detection_Number": "N/A",
                            "Confidence_Score": "N/A",
                            "X1_Coordinate": "N/A",
                            "Y1_Coordinate": "N/A", 
                            "X2_Coordinate": "N/A",
                            "Y2_Coordinate": "N/A",
                            "Width_Pixels": "N/A",
                            "Height_Pixels": "N/A",
                            "Center_X": "N/A",
                            "Center_Y": "N/A"
                        })
                    else:
                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            width = x2 - x1
                            height = y2 - y1
                            center_x = x1 + width // 2
                            center_y = y1 + height // 2
                            
                            table_data.append({
                                "Detection_ID": f"DET_{detection_id:04d}",
                                "Image_Name": image_name,
                                "Ant_Count": len(boxes),
                                "Detection_Number": i + 1,
                                "Confidence_Score": f"{score:.4f}",
                                "X1_Coordinate": x1,
                                "Y1_Coordinate": y1,
                                "X2_Coordinate": x2,
                                "Y2_Coordinate": y2,
                                "Width_Pixels": width,
                                "Height_Pixels": height,
                                "Center_X": center_x,
                                "Center_Y": center_y
                            })
                            detection_id += 1
                
                # Write to CSV
                import csv
                if table_data:
                    fieldnames = table_data[0].keys()
                    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(table_data)
                
                # Create summary file
                summary_filepath = filepath.replace('.csv', '_summary.txt')
                batch_summary = self.prediction_results["batch_summary"]
                parameters = self.prediction_results["parameters"]
                
                summary_content = f"""Ant Detection Results Summary
================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BATCH SUMMARY
-------------
Total Images Processed: {batch_summary['processed_images']}
Total Ants Detected: {batch_summary['total_ants']}
Average Ants per Image: {batch_summary['total_ants'] / max(batch_summary['processed_images'], 1):.2f}
Processing Time: {batch_summary['processing_time']:.2f} seconds

DETECTION PARAMETERS
--------------------
Confidence Threshold: {parameters['confidence_threshold']:.3f}
Distance Threshold: {parameters['distance_threshold']:.3f}
Distance Method: {'Normalized' if parameters['use_normalized_distance'] else 'Pixel-based'}
Fast Detection Mode: {parameters['fast_mode']}
Show All Detections: {parameters['show_all_detections']}

PER-IMAGE SUMMARY
-----------------
"""
                
                for result in all_results:
                    summary_content += f"{result['image_name']}: {result['ant_count']} ants detected\n"
                
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    f.write(summary_content)
                
                messagebox.showinfo("Export Successful", 
                                  f"Detection results exported successfully:\n\n"
                                  f"Detailed CSV: {os.path.basename(filepath)}\n"
                                  f"Summary: {os.path.basename(summary_filepath)}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")

    def create_prediction_results_panel(self, parent, row, column):
        """Create prediction results panel with summary metrics and image display"""
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding="5")
        results_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(2, weight=1)
        
        # Summary metrics frame
        summary_frame = ttk.LabelFrame(results_frame, text="Batch Summary")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create summary metric labels in a grid
        self.pred_summary_metrics = {}
        summary_names = ['Images Processed', 'Total Ants Found', 'Average per Image', 'Processing Time']
        
        for i, name in enumerate(summary_names):
            # Create label and value in a 2x2 grid
            row_pos = i // 2
            col_pos = (i % 2) * 2
            
            label = ttk.Label(summary_frame, text=f"{name}:", style='Info.TLabel')
            label.grid(row=row_pos, column=col_pos, sticky=tk.W, padx=10, pady=2)
            
            value_label = ttk.Label(summary_frame, text="--", style='Info.TLabel', font=('Arial', 9, 'bold'))
            value_label.grid(row=row_pos, column=col_pos+1, sticky=tk.W, padx=5, pady=2)
            self.pred_summary_metrics[name.lower().replace(' ', '_')] = value_label
        
        # Current image info frame
        info_frame = ttk.Frame(results_frame)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.current_image_label = ttk.Label(info_frame, text="No image selected", style='Info.TLabel')
        self.current_image_label.pack(side=tk.LEFT)
        
        self.detection_count_label = ttk.Label(info_frame, text="Detections: 0", style='Info.TLabel')
        self.detection_count_label.pack(side=tk.RIGHT)
        
        # Image display frame with canvas
        canvas_frame = ttk.LabelFrame(results_frame, text="Current Detection Result")
        canvas_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create scrollable canvas
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        # Scrollbars for canvas
        h_scroll_pred = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll_pred = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        
        # Canvas for prediction display
        self.pred_canvas = tk.Canvas(canvas_container, bg='white', height=400,
                                   xscrollcommand=h_scroll_pred.set,
                                   yscrollcommand=v_scroll_pred.set,
                                   highlightthickness=2, highlightbackground=self.colors['accent'])
        
        h_scroll_pred.config(command=self.pred_canvas.xview)
        v_scroll_pred.config(command=self.pred_canvas.yview)
        
        # Grid layout for canvas and scrollbars
        self.pred_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scroll_pred.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scroll_pred.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initialize display
        self.pred_canvas.create_text(200, 100, text="Load images and run detection to see results", 
                                   fill="gray", font=("Arial", 12))

    def export_detection_summary(self):
        """Export summary detection results - one row per image"""
        if not hasattr(self, 'prediction_results') or not self.prediction_results:
            messagebox.showwarning("Warning", "No prediction results to export. Please run detection first.")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ant_detection_summary_{timestamp}.csv"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            title="Save Detection Summary",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                all_results = self.prediction_results["all_results"]
                
                # Create summary data - one row per image
                summary_data = []
                headers = ["Image_Name", "Total_Ants_Detected", "Average_Confidence", "Min_Confidence", "Max_Confidence"]
                summary_data.append(headers)
                
                for result in all_results:
                    image_name = result["image_name"]
                    ant_count = result["ant_count"]
                    scores = result["scores"]
                    
                    if len(scores) > 0:
                        avg_conf = sum(scores) / len(scores)
                        min_conf = min(scores)
                        max_conf = max(scores)
                    else:
                        avg_conf = min_conf = max_conf = 0
                    
                    row = [image_name, ant_count, f"{avg_conf:.4f}", f"{min_conf:.4f}", f"{max_conf:.4f}"]
                    summary_data.append(row)
                
                # Write to CSV
                import csv
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(summary_data)
                
                messagebox.showinfo("Export Successful", f"Summary results exported to:\n{filepath}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting summary: {str(e)}")

    def export_detection_detailed(self):
        """Export detailed detection results - one row per detection"""
        if not hasattr(self, 'prediction_results') or not self.prediction_results:
            messagebox.showwarning("Warning", "No prediction results to export. Please run detection first.")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ant_detection_detailed_{timestamp}.csv"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            title="Save Detailed Detection Results",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Use the existing detailed export logic
                all_results = self.prediction_results["all_results"]
                
                # Create comprehensive results table
                table_data = []
                detection_id = 1
                
                headers = ["Detection_ID", "Image_Name", "Detection_Number", "Confidence_Score", 
                          "X1_Coordinate", "Y1_Coordinate", "X2_Coordinate", "Y2_Coordinate",
                          "Width_Pixels", "Height_Pixels", "Center_X", "Center_Y"]
                table_data.append(headers)
                
                for result in all_results:
                    image_name = result["image_name"]
                    boxes = result["boxes"]
                    scores = result["scores"]
                    
                    if len(boxes) == 0:
                        # Add row for images with no detections
                        row = [f"IMG_{len([r for r in all_results if r['ant_count'] == 0]):03d}", 
                               image_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
                        table_data.append(row)
                    else:
                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            width = x2 - x1
                            height = y2 - y1
                            center_x = x1 + width // 2
                            center_y = y1 + height // 2
                            
                            row = [f"DET_{detection_id:04d}", image_name, i + 1, f"{score:.4f}",
                                  x1, y1, x2, y2, width, height, center_x, center_y]
                            table_data.append(row)
                            detection_id += 1
                
                # Write to CSV
                import csv
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(table_data)
                
                messagebox.showinfo("Export Successful", f"Detailed results exported to:\n{filepath}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting detailed results: {str(e)}")

    # Utility functions
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
        except:
            self.memory_label.config(text="Memory: -- MB")
        
        # Schedule next update
        self.root.after(2000, self.update_memory_usage)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AntDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()