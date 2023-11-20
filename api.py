import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from architectures.Architecture import MyArchitecture,Generator
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import io
import cv2
import numpy as np
from torchvision.utils import save_image
from scipy.stats import truncnorm
import torch.optim as optim

app = FastAPI()

model_classify = MyArchitecture(5)
model_classify.load_state_dict(torch.load(r"model/classification.pth", map_location=torch.device('cpu')))
model_YOLO_detection = YOLO(r"model/YOLOv8detection.pt")
model_YOLO_segmentation = YOLO(r"model/YOLOv8segmentation.pt")
model_generator_all = torch.load('model/generator_all.pth', map_location="cpu")
model_generator_stego = torch.load('model/generator_stego.pth', map_location="cpu")
def classify_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform_classify = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform_classify(image)
    input_batch = input_tensor.unsqueeze(0)
    class_to_idx = {'trex': 0, 'para': 1, 'stego': 2, 'velo': 3, 'spino': 4}

    with torch.no_grad():
        output = model_classify(input_batch)

    return image, [i for i in class_to_idx if class_to_idx[i]==int(torch.argmax(output))]

def detect_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model_YOLO_detection(image)
    names = model_YOLO_detection.names
    for r in results:
        boxes = r.boxes
        for box in boxes:  # iterate boxes
            r = box.xyxy[0].numpy()  # get corner points as int
        for c in boxes.cls:
            predicted_class = names[int(c)]
    return image,list(r),predicted_class

def segmentation_image(image_bytes):

    img = np.array(Image.open(io.BytesIO(image_bytes)))
    H, W, _ = img.shape
    img = cv2.resize(img, (384, 640))
    results = model_YOLO_segmentation(img)
    result = results[0]
    masks = result.masks
    polygon = masks.xy[0]

    return img,polygon

def generate_examples(gen, steps, truncation=0.7, n=10):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, 256, 1, 1)), device=torch.device('cpu'), dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, rf"C:\Users\mirok\PycharmProjects\API\generated\img_{i}.jpg")



@app.post("/classify_all/", description="This endpoint takes as input image a return image with predicted class")
async def classify_all_endpoint(image: UploadFile = File(...)):

    image_bytes = await image.read()
    classified_image, predicted_class = classify_image(image_bytes)
    output_image_path = "classified_image.jpg"
    classified_image.save(output_image_path)
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    img = cv2.imread(output_image_path)
    cv2.putText(img,str(predicted_class),position, font, font_scale, font_color, font_thickness)
    cv2.imwrite(output_image_path, img)

    return FileResponse(output_image_path, media_type="image/jpeg")

@app.post("/detect_all/", description="This endpoint takes as input image a return image with predicted class and bounding box")
async def detect_all_endpoint(image: UploadFile = File(...)):

    image_bytes = await image.read()
    detected_image,predicted_boxes,predicted_class = detect_image(image_bytes)
    output_image_path = "detected_image.jpg"
    detected_image.save(output_image_path)
    color = (0, 255, 0)
    thickness = 2
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    img = cv2.imread(output_image_path)
    cv2.rectangle(img,(int(predicted_boxes[0]), int(predicted_boxes[1])), (int(predicted_boxes[2]), int(predicted_boxes[3])), color,thickness)
    cv2.putText(img,str(predicted_class),position, font, font_scale, font_color, font_thickness)
    cv2.imwrite(output_image_path, img)
    return FileResponse(output_image_path, media_type="image/jpeg")

@app.post("/segmentation_stego/", description="This endpoint works just for stego class due long preparation of data. This endpoint takes as input image and returns mask as image of predicted class")
async def segmentation_stego_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img,polygon = segmentation_image(image_bytes)
    output_image_path = "segmentation_image.jpg"
    cv2.polylines(img, np.int32([polygon]), isClosed=True, color=(255, 0, 0), thickness=5)
    cv2.imwrite(output_image_path, img)
    return FileResponse(output_image_path, media_type="image/jpeg")

@app.post("/generate_all/",description="This endpoint is in progress! For image resolution use int in range 0-8. 0 -> 4x4 1 ->8x8 2 -> 16x16 ....")
async def generate_all_endpoint(image_resolution:int ):
    output_image_path="generated_images.jpg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    CHANNELS_IMG = 3
    Z_DIM = 256
    IN_CHANNELS = 256
    gen = Generator(
        Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.load_state_dict(model_generator_all["state_dict"])
    opt_gen.load_state_dict(model_generator_all["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = LEARNING_RATE
    generate_examples(gen, steps=image_resolution, truncation=0.7, n=10)
    image_list = []
    path_to_image_folder = "generated"
    for i in os.listdir(path_to_image_folder):
        image_list.append(cv2.imread(os.path.join(path_to_image_folder, i)))
    result = np.vstack(image_list)
    cv2.imwrite('generated_images.jpg', result)

    return FileResponse(output_image_path, media_type="image/jpeg")
@app.post("/generate_stego/",description="This endpoint is in progress! For image resolution use int in range 0-8. 0 -> 4x4 1 ->8x8 2 -> 16x16 ....")
async def generate_all_endpoint(image_resolution:int ):
    output_image_path="generated_images.jpg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    CHANNELS_IMG = 3
    Z_DIM = 256
    IN_CHANNELS = 256
    gen = Generator(
        Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.load_state_dict(model_generator_stego["state_dict"])
    opt_gen.load_state_dict(model_generator_stego["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = LEARNING_RATE
    generate_examples(gen, steps=image_resolution, truncation=0.7, n=10)
    image_list = []
    path_to_image_folder = "generated"
    for i in os.listdir(path_to_image_folder):
        image_list.append(cv2.imread(os.path.join(path_to_image_folder, i)))
    result = np.vstack(image_list)
    cv2.imwrite('generated_images.jpg', result)

    return FileResponse(output_image_path, media_type="image/jpeg")
@app.get("/")
def root():
    return "Welcome to API for using api go to /docs"

