import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ========================
# PATH
# ========================
base_dir = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images"

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Optimal for EfficientNetB3
IMG_SIZE = 300
BATCH_SIZE = 16 # Reduced batch size for B3 on CPU/Low RAM

# ========================
# RESULTS FOLDER
# ========================
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# ========================
# ADVANCED DATA AUGMENTATION (DYNAMIC)
# ========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],
    validation_split=0.2 # Use 20% of train folder for internal validation
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========================
# SAVE LABELS
# ========================
class_map = train_gen.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_map, f)

num_classes = len(class_map)
print("Classes:", num_classes)

# ========================
# CLASS WEIGHT
# ========================
classes = np.unique(train_gen.classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ========================
# MODEL ARCHITECTURE
# ========================
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Initially freeze the base
base_model.trainable = False

# Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x) # Increased size
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ========================
# LOSS & COMPILE
# ========================
# Label Smoothing helps with regularization and reaching higher accuracy
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ========================
# CALLBACKS
# ========================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1, min_lr=1e-7)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# ========================
# PHASE 1: HEAD WARMUP
# ========================
print("\nPhase 1: Training Head (Base Frozen)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy']
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15, # Shorter warmup
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# ========================
# PHASE 2: PARTIAL FINE-TUNING
# ========================
print("\nPhase 2: Partial Fine-tuning (Top 30% Unfrozen)...")
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR
    loss=loss_fn,
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# ========================
# PHASE 3: FULL FINE-TUNING
# ========================
print("\nPhase 3: Full Fine-tuning (All Layers Unfrozen)...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very Low LR
    loss=loss_fn,
    metrics=['accuracy']
)

history3 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# ========================
# FINAL RESULTS & TEST
# ========================
print("\nTRAINING RESULT")
model.save("final_model.h5")

print("\nTESTING ON TEST SET...")
test_loss, test_acc = model.evaluate(test_gen)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# ========================
# CONFUSION MATRIX
# ========================
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15,12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Final Confusion Matrix')
cm_path = os.path.join(results_dir, "confusion_matrix_final.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# Report
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(results_dir, "classification_report_final.txt"), "w") as f:
    f.write(report)

print(f"Reports saved in {results_dir}")