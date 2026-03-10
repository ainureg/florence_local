import cv2
from flor import run_florence

img = cv2.imread("/home/ainur/photo.jpg")
if img is None:
    raise ValueError

r = run_florence(img)
print(r["clean_text"])
