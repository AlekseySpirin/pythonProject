from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/spiri/PycharmProjects/pythonProject/data/tableware_dataset',
            epochs=1, imgsz=64, labels='labels')
