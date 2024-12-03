import time
import psutil
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("/Codings/Husk_grading/Scripts/runs/detect/train4/weights/best.pt")  # Path to the best-trained model

# Measure initial memory usage
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
print(f"Initial Memory Usage: {initial_memory:.2f} MB")

# Measure initial CPU usage
initial_cpu = psutil.cpu_percent(interval=0.1)
print(f"Initial CPU Usage: {initial_cpu:.2f}%")

# Start tracking inference time
start_time = time.time()

# Test the model on a single image
results = model.predict(source="/SLIIT/Research/Test/test2.jpg", save=True, imgsz=640)

# End tracking inference time
end_time = time.time()

# Calculate inference time
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# Measure memory usage after inference
final_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
print(f"Memory Usage After Inference: {final_memory:.2f} MB")

# Measure CPU usage after inference
final_cpu = psutil.cpu_percent(interval=0.1)
print(f"CPU Usage After Inference: {final_cpu:.2f}%")

# Analyze results
detections = results[0].boxes if results else None  # Get bounding box details

if detections:
    print(f"Total Detections: {len(detections)}")
    for i, detection in enumerate(detections):
        # Extract confidence, class, and bounding box coordinates
        confidence = detection.conf.item()  # Confidence score
        class_id = detection.cls.item()  # Class ID
        box = detection.xyxy[0].tolist()  # Bounding box (x_min, y_min, x_max, y_max)

        print(f"Detection {i + 1}:")
        print(f"  - Class ID: {class_id}")
        print(f"  - Confidence: {confidence:.4f}")
        print(f"  - Bounding Box: {box}")
else:
    print("No detections were made.")

# Save results for further analysis (optional)
results.save_txt()

# Calculate memory and CPU usage differences
memory_usage = final_memory - initial_memory
cpu_usage = final_cpu - initial_cpu

print(f"Memory Increase During Inference: {memory_usage:.2f} MB")
print(f"CPU Usage During Inference: {cpu_usage:.2f}%")
