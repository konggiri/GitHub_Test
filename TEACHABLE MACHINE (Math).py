import tensorflow.keras
from PIL import Image
import numpy as np
import cv2



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


#카메라 설정
video_capture_0 = cv2.VideoCapture(0) #노트북 내장 웹캠

while True:
    ret0, frame0 = video_capture_0.read()
# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
    if(ret0):
        
        cv2.imwrite('capture.png',frame0, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        #print("SAVE")
        image = Image.open('capture.png')
        
        image = image.resize((224, 224))
        image_array = np.asarray(image)

# Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
        data[0] = normalized_image_array

# run the inference
        prediction = model.predict(data)
        print(prediction[0][0])
        

        
        #a=str(prediction[0])
        #print(a.split(' ')[0])
video_capture_0.release()
cv2.destroyAllWindows() # 리소스 반환
#개발환경 구축
#pip install numpy==1.16.4
# pip3 install pillow 
# git clone https://github.com/fchollet/keras.git
# pip install tensorflow==2.0.0-beta1
# pip install --upgrade absl-py
