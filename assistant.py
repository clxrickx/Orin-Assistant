import cv2                  # Camera and image display
import numpy as np          # Array manipulation
import torch                # AI model computation
from torchvision import models, transforms  # Pretrained vision models
import sounddevice as sd  # Audio i/o
import time                 
import os
import psutil # System monitoring
from coco_labels import COCO_CLASSES  # COCO dataset labels


frames_loaded = 0
AUDIO_THRESHOLD = 0.05   #audio sens.
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(cv2.CAP_PROP_FPS, 60)  # Frame rate



def clr():
    os.system('clear')

def detect_sound(duration=0.1, samplerate=44100):
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    volume = np.linalg.norm(audio)
    return volume > AUDIO_THRESHOLD, volume

def print_detected_objects(labels, scores, threshold=0.3):
    detected = []
    for label, score in zip(labels, scores):
        if score > threshold:
            detected.append(f"{COCO_CLASSES.get(int(label), 'unknown')}: {score:.2f}")
    if detected:
        print("Detected objects:\n", "\n".join(detected))
    else:
        print("No objects detected above threshold ", threshold*100, "%")

clr()
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera opened successfully")


#SSD faster but less accurate
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
model = ssdlite320_mobilenet_v3_large(pretrained=True)
#ai model, Faster R-CNN is accurate but slow: use SSD for faster stuff:
#from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
#weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
print("Model loaded successfully.")

# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor()  # Converts HxWxC NumPy array (0-255) to CxHxW tensor (0-1)
])
# Main loop

while True:
    cpu_percent = psutil.cpu_percent(interval=None) #CPU load
    mem = psutil.virtual_memory() #RAM load
    ram_used = mem.percent
    sound_detected, volume = detect_sound()
    frames_loaded += 1
    # 5a. Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 5b. Convert frame to PyTorch tensor
    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]

    # 5c. Run AI model (inference only, no gradients)
    with torch.no_grad():
        outputs = model(input_tensor)

    # 5d. Extract model outputs
    boxes = outputs[0]['boxes']       # Bounding boxes [x1, y1, x2, y2]
    labels = outputs[0]['labels']     # Object class labels (person = 1)
    scores = outputs[0]['scores']     # Confidence scores (0-1)

    # Measure inference time/alt to TOPS
    start = time.time()

    with torch.no_grad():
        outputs = model(input_tensor)

    inference_time = time.time() - start


    # Draw detections on the frame, rect.
    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > 0.8:  # person with high confidence
            x1, y1, x2, y2 = box.int().tolist()

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),   # bgr color
                5              # thickness
            )

            # Draw label text
            class_name = COCO_CLASSES.get(int(label), "unknown")
            text = f"{class_name}: {score:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        if sound_detected:
            cv2.putText(
                frame,
                "SOUND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),
                4
            )


    person_detected = False
    for label, score in zip(labels, scores):
        if label == 1 and score > 0.8: #confidence of 0.8
            person_detected = True
            break
    clr()
    print("Person detected:", person_detected)
    print("Frames loaded:", frames_loaded)
    print("Sound detected:", sound_detected, "Volume:", round(volume, 3))
    print("CPU Load (%):", cpu_percent)
    print("RAM Used (%):", ram_used)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
