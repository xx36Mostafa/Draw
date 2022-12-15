import cv2 
from sketchpy import canvas
from sklearn.preprocessing import scale

pic = canvas.sketch_from_svg(r"photo.jpg")
pic.draw()
photo = cv2.imread(r"photo.png")
photo_grey = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
photo_invert = cv2.bitwise_not(photo_grey) 
photo_blur = cv2.GaussianBlur(photo_invert,(21,21),0)
photo_invblur = cv2.bitwise_not(photo_blur)
sketch = cv2.divide(photo_grey,photo_invblur,scale=256.0)
cv2.imwrite("sketch.png",sketch)