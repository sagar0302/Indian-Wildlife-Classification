# train_cnn_svm_resnet_pro.py
import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler # <--- NEW
from sklearn.model_selection import GridSearchCV # <--- NEW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from PIL import ImageFile

# 1. SET GLOBAL SEEDS (Reproducibility)
np.random.seed(42)
tf.random.set_seed(42)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----- CONFIG -----
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

PROJECT_DIR = r"C:\Users\sagar\CDAC_Mini Project"
DATA_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")

# File paths
FEATURES_PATH = os.path.join(PROJECT_DIR, "cnn_features_resnet_final.npz")
SVM_MODEL_PATH = os.path.join(PROJECT_DIR, "svm_model_resnet_final.joblib")
CNN_MODEL_PATH = os.path.join(PROJECT_DIR, "cnn_feature_extractor_resnet_final.h5")
HEATMAP_PATH = os.path.join(PROJECT_DIR, "confusion_matrix_heatmap_final.png")

# ----- DATA GENERATORS -----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,      # Increased slightly
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def safe_flow_from_directory(datagen, directory, **kwargs):
    try:
        return datagen.flow_from_directory(directory, **kwargs)
    except Exception as e:
        print("Error reading image:", e)
        return None

# --- STEP 1: CLASS WEIGHTS ---
print("Scanning training data for class weights...")
temp_gen = safe_flow_from_directory(train_datagen, TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False)
classes = np.unique(temp_gen.classes)
class_weights_array = compute_class_weight('balanced', classes=classes, y=temp_gen.classes)
class_weights_dict = dict(enumerate(class_weights_array))

# --- STEP 2: GENERATORS ---
train_gen = safe_flow_from_directory(train_datagen, TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="sparse", shuffle=True, seed=42)
val_gen = safe_flow_from_directory(val_datagen, VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False, seed=42)

num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())

# ----- STEP 3: ROBUST MODEL ARCHITECTURE -----
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False 

x = layers.GlobalAveragePooling2D()(base_model.output)
# IMPROVEMENT: Add Dropout to prevent overfitting
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)  # <--- Discards 50% of neurons randomly during training
feature_extractor = models.Model(inputs=base_model.input, outputs=x, name="resnet50_feature_extractor")

x_clf = layers.Dense(num_classes, activation="softmax")(feature_extractor.output)
clf_model = models.Model(inputs=feature_extractor.input, outputs=x_clf)

clf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# ----- STEP 4: TRAIN CNN -----
print("\nStarting CNN Training...")
clf_model.fit(train_gen, epochs=25, validation_data=val_gen, callbacks=[early_stop], class_weight=class_weights_dict)

# ----- STEP 5: EXTRACT FEATURES -----
# We take the output of the Dense(256) layer (before dropout)
feature_extractor_only = models.Model(inputs=clf_model.input, outputs=clf_model.layers[-3].output) 

def extract_features(generator, model):
    print("Extracting features...")
    features, labels = [], []
    generator.reset()
    for i in range(len(generator)):
        x_batch, y_batch = next(generator)
        if x_batch.shape[0] != y_batch.shape[0]: break
        f = model.predict(x_batch, verbose=0)
        features.append(f)
        labels.append(y_batch)
    return np.concatenate(features), np.concatenate(labels)

train_gen_no_shuffle = safe_flow_from_directory(val_datagen, TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False)

train_features, train_labels = extract_features(train_gen_no_shuffle, feature_extractor_only)
val_features, val_labels = extract_features(val_gen, feature_extractor_only)

# IMPROVEMENT: Scale Features
print("Scaling features...")
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)

# ----- STEP 6: SVM WITH GRID SEARCH -----
print("\nOptimizing SVM...")
# IMPROVEMENT: Grid Search to find best parameters
param_grid = {
    'C': [0.1, 1, 10], 
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
# We still use class_weight='balanced'
grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, refit=True, verbose=2)
grid.fit(train_features_scaled, train_labels)

best_svm = grid.best_estimator_
print(f"Best SVM Parameters: {grid.best_params_}")

# ----- EVALUATION -----
val_pred = best_svm.predict(val_features_scaled)

print("\nClassification Report:")
print(classification_report(val_labels, val_pred, target_names=class_names))

cm = confusion_matrix(val_labels, val_pred)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Optimized SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(HEATMAP_PATH)
plt.show()

# SAVE EVERYTHING (Include Scaler!)
joblib.dump({
    "svm": best_svm, 
    "class_indices": train_gen.class_indices,
    "scaler": scaler  # <--- IMPORTANT: Save scaler for prediction time
}, SVM_MODEL_PATH)
feature_extractor_only.save(CNN_MODEL_PATH)
print(f"\nPro-Grade Models saved to {PROJECT_DIR}")