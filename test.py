from ultralytics import YOLO
model=YOLO("best.pt")
results=model("test_sample.jpg" ,show=True)
results[0].show
results[0].save(filename="detected_output.jpg")
