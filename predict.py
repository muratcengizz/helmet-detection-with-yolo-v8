from ultralytics import YOLO
import os
import cv2
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO("C:/Users/murat/runs/detect/train3/weights/best.pt")

path = os.chdir("C:/Users/murat/OneDrive/Belgeler/computerVision/yoloPractice/hard_hat_sample/test/images")
path = os.listdir(path)


for image in path:
    img = cv2.imread(filename=image)
    #img = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
    
    results = model.predict(img)
    
    plotted = results[0].plot()
    cv2.imshow(winname="detection", mat=plotted)
    if cv2.waitKey(0) == ord("q"):
        cv2.imwrite(filename=image, img=plotted)
    
    
cv2.destroyAllWindows()



    