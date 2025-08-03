from ultralytics import YOLO
import kagglehub

# Download latest version
path = kagglehub.dataset_download("pavansanagapati/images-dataset")

print("Path to dataset files:", path)


# Load a model
model = YOLO("yolo11m.pt")  # pretrained YOLO11n model

sample_image01 = path + "/data/cars/carsgraz_391.bmp"
sample_image02 = path + "/data/cars/carsgraz_390.bmp"

# Run batched inference on a list of images
results = model([sample_image01, sample_image02])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk