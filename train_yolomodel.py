from ultralytics import YOLO
import time

# === 1. Load base YOLO model ===
model = YOLO("yolov8n.pt")  # or yolo11n.pt if you have it locally

# === 2. Train ===
t0 = time.time()
results = model.train(
    data="Basketball.v1i.yolov11/data.yaml",   # your dataset YAML
    epochs=20,
    imgsz=416,
    batch=32,
    device="cpu",          # or "cpu"
    patience=50,
    augment=False,
    mosaic=False,
    mixup=False,
    project="output",      # ← base directory for results
    name="basketball_run", # ← subfolder name inside project
)
train_time = time.time() - t0
print(f"\n✅ Training completed in {train_time/60:.2f} minutes")

# === 3. Evaluate on test split ===
metrics = model.val(data="./params.yaml", split="test", imgsz=416)

# === 4. Extract and print metrics ===
try:
    precision = metrics.results_dict.get("metrics/precision(B)", None)
    recall = metrics.results_dict.get("metrics/recall(B)", None)
    f1 = 2 * (precision * recall) / (precision + recall) if precision and recall else None
except Exception:
    precision = recall = f1 = None

print("\n📊 Evaluation Results:")
print(f" Precision: {precision:.4f}" if precision is not None else " Precision: (not available)")
print(f" Recall:    {recall:.4f}" if recall is not None else " Recall: (not available)")
print(f" F1 Score:  {f1:.4f}" if f1 is not None else " F1 Score: (not computed)")
print(f" mAP@50:    {metrics.box.map50:.4f}")
print(f" mAP@50-95: {metrics.box.map:.4f}")

# === 5. Save best model path ===
print(f"\n🏁 Best model saved at: {model.ckpt_dir}/weights/best.pt")

# === 6. (Optional) Test inference on sample images ===
# model.predict(source="images/val/", save=True)
