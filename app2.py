import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ---------------- Paths ----------------
disease_model_path = os.path.join("model", "model.keras")
dataset_dir = "/Users/rithwik/Downloads/plant-disease-backend 4/Final_Dataset"
classes_path = os.path.join("model", "label_classes.npy")

# ---------------- Load model and classes ----------------
disease_model = load_model(disease_model_path)
classes = np.load(classes_path, allow_pickle=True).tolist()  # 15 classes

# ---------------- Data generator ----------------
test_datagen = ImageDataGenerator(rescale=1./255)

data_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=classes  # Only include classes your model knows
)

# ---------------- Get predictions ----------------
preds = disease_model.predict(data_generator)
pred_labels = np.argmax(preds, axis=1)  # Predicted class indices
true_labels = data_generator.classes    # True class indices

# ---------------- Map class indices to names ----------------
class_indices = data_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

# ---------------- Calculate per-class accuracy ----------------
per_class_correct = {class_name: 0 for class_name in classes}
per_class_total = {class_name: 0 for class_name in classes}

for true, pred in zip(true_labels, pred_labels):
    class_name = inv_class_indices[true]
    per_class_total[class_name] += 1
    if true == pred:
        per_class_correct[class_name] += 1

print("Per-class accuracy:")
for class_name in classes:
    total = per_class_total[class_name]
    correct = per_class_correct[class_name]
    if total > 0:
        acc = correct / total * 100
        print(f"{class_name}: {acc:.2f}% ({correct}/{total})")
    else:
        print(f"{class_name}: No images found in dataset")
