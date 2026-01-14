import cv2                  # Camera and image display
import numpy as np          # Array manipulation
import torch                # AI model computation
from torchvision import models, transforms  # Pretrained vision models
import sounddevice as sd  # Audio i/o
import time                 
import os
import psutil # System monitoring
from coco_labels import COCO_CLASSES  # COCO dataset labels

batch_size = 0
frames_loaded = 0
AUDIO_THRESHOLD = 0.2   #audio sens, higher will increase performance, less overhead
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(cv2.CAP_PROP_FPS, 60)  # Frame rate

def clr():
    os.system('cls' if os.name == 'nt' else 'clear')

def detect_sound(duration=0.1, samplerate=22050):
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    volume = np.linalg.norm(audio)
    return volume > AUDIO_THRESHOLD, volume

clr()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available. Using GPU. Thanks NVIDIA :)')
    print("GPU: ", torch.cuda.get_device_name(0))
    batch_size = 3  # Increased batch size for faster inference on GPU
else:
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("MPS is available. Using MPS device. Thanks Apple :)")
    #else:
    device = torch.device("cpu")
    print("WARNING: MPS is not available. Falling back to CPU. Puny Mac user :(")
    batch_size = 1 #slower inference for those puny mac users lol (me)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera opened successfully")

#SSD faster but less accurate
#from torchvision.models.detection import ssdlite320_mobilenet_v3_large
#model = ssdlite320_mobilenet_v3_large(pretrained=True)
#ai model, Faster R-CNN is accurate but slow: use SSD for faster stuff:
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
model.to(device)
print("Model loaded successfully.")
input("Press Enter to continue...")
transform = transforms.Compose([
    transforms.ToTensor()  # Converts HxWxC NumPy array (0-255) to CxHxW tensor (0-1)
])


while True:
    cpu_percent = psutil.cpu_percent(interval=None) #CPU load
    mem = psutil.virtual_memory() #RAM load
    ram_used = mem.percent
    frames_loaded += 1

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    input_tensor = input_tensor.to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
    inference_time = time.time() - start

    boxes = outputs[0]['boxes']       # Bounding boxes [x1, y1, x2, y2]
    labels = outputs[0]['labels']     # Object class labels (person = 1)
    scores = outputs[0]['scores']     # Confidence scores (0-1)
    cv2.putText(frame, f"CPU Load: {cpu_percent}% | RAM Used: {ram_used}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 4)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.8:  # person with high confidence
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            class_name = COCO_CLASSES.get(int(label), "unknown")
            cv2.putText(frame, class_name + f": {score*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if score < 0.8 and score > 0.3:  # objects with low confidence
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
            class_name = COCO_CLASSES.get(int(label), "unknown")
            cv2.putText(frame, class_name + f": {score*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

#removed sound detection for now, causes unnecessary overhead

    clr()
    print("Frames loaded:", frames_loaded)
    print("Sound detected:", sound_detected, "Volume:", round(volume, 3))
    print("Inference time:", round(inference_time, 3), "seconds")    

    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
