"""
https://docs.ultralytics.com/modes/predict/#inference-sources
"""

from ultralytics import YOLO

# Load a model
model = YOLO('bbox-yolo/yolov8n.pt')  # pretrained YOLOv8n model
# model = YOLO('yolov8n.pt') 
# model = YOLO('yolov8n.pt', cfg="cfg/default.yaml")  # pretrained YOLOv8n model

# Run batched inference on a list of images
#results = model(['datasets/my_examples/img1.png', 'datasets/my_examples/img2.png', 'datasets/my_examples/img3.png'], save_dir="bbox-yolo/save_dir")  # return a list of Results objects


#results = model(['datasets/my_examples/img1.png', 'datasets/my_examples/img2.png', 'datasets/my_examples/img3.png'], save_dir = "/cluster/work/cotterell/yemara/ml-4mbbox-yolo/save_dir")  # return a list of Results objects
#save_dir="/cluster/work/cotterell/yemara/predict_results"

results = model(
    ['/store/swissai/a08/yemara/ml-4m/bbox-yolo/datasets/my_examples/img1.png', '/store/swissai/a08/yemara/ml-4m/bbox-yolo/datasets/my_examples/img2.png', '/store/swissai/a08/yemara/ml-4m/bbox-yolo/datasets/my_examples/img3.png'],
    project='/store/swissai/a08/yemara/ml-4m/bbox-yolo/save_dir', 
    name='testing_prediction'
)

print("PASSED MODEL")
# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    # import pdb; pdb.set_trace()
    #result.show()  # display to screen
    result.save(filename=f'result{i+1}.jpg')  # save to disk