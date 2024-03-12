import numpy as np
from ultralytics import YOLO

model = YOLO('./runs/classify/train10/weights/last.pt')  # load a custom model

results = model('./data/tableware_dataset/images/train/fork/fork1.jpg')  # predict on an image

print(results)

names_dict = results[0].names

probs = results[0].probs.tolist()


print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])
