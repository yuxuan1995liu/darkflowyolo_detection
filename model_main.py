import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

import pprint as pp
import os

# options = {"model": "cfg/custum_yolo.cfg",
#           "batch": 8,
#             "load": "bin/yolo.weights",
#            "epoch": 3,
#            "trainer":"adam",
#            "gpu": 1.0,
#            "train": True,
#            "annotation": "train/train_anno/",
#            "dataset": "train/train_img/"}
# tfnet = TFNet(options)
#tfnet.load_from_ckpt()


options = {"model": "cfg/custum_yolo.cfg", 
           "load": -1,
           "batch": 16,
           "epoch": 4,
           "gpu": 1.0,
           "train": True,
           "annotation": "train/train_anno/",
           "dataset": "train/train_img/"}

tfnet = TFNet(options)
# #tfnet.load_from_ckpt()
tfnet.train()
tfnet.savepb()

#prediction
img_names = os.listdir('test/test_img/')
cnt_valid = 0
for names in img_names:
  if names[-1] =='g':
    original_img = cv2.imread("test/test_img/"+names)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(original_img)
    print(results)
    if results !=[]:
      cnt_valid+=1
print(cnt_valid)
# original_img = cv2.imread("test/test_img/00029843_001.png")
# original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
# results = tfnet.return_predict(original_img)

def boxing(original_img , predictions):
    newImage = np.copy(original_img)
    for result in predictions:

        print(result)
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']
        print(top_x,top_y)

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']
        print(btm_x,btm_y)

        confidence = result['confidence']
        print(confidence)
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.1:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage
# fig, ax = plt.subplots(figsize=(20, 10))
# ax.imshow(boxing(original_img, results))

# plt.show()