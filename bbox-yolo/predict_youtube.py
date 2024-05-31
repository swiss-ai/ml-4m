from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define source as YouTube video URL
source = 'https://youtu.be/HPhYWxiOSB4'  #LNwODJXcvt4'

# Run inference on the source
results = model(source, vid_stride=5, stream=True)  # generator of Results objects

# Visualize results
for i, result in enumerate(results):
    result.show()  # display to screen
    result.save(filename=f'result_eggs_{i}.jpg')  # save to disk