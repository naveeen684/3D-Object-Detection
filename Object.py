from imageai.Detection import ObjectDetection
detector = ObjectDetection()
model_path = "model/yolo-tiny.h5"
input_path = "data/1.jpg"
output_path = "Output/1.jpg"
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
