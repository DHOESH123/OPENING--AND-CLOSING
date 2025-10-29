# Implementation-of-Opening-and-Closing
###  NAME :ESHWAR T
### REG NO :212223230054
## Aim
To implement Opening and Closing morphological operations using Python and OpenCV.

## Software Required
1. Anaconda - Python 3.7  
2. OpenCV  

## Algorithm:
### Step1:
Import the necessary libraries (OpenCV, NumPy, Matplotlib).

### Step2:
Create a blank black image using NumPy and add text using `cv2.putText()`.

### Step3:
Create a structuring element (kernel) using `np.ones()`.

### Step4:
For **Opening**:
- Add white noise to the image.
- Apply `cv2.morphologyEx()` with `cv2.MORPH_OPEN` to remove background noise.

### Step5:
For **Closing**:
- Add black noise to the image.
- Apply `cv2.morphologyEx()` with `cv2.MORPH_CLOSE` to remove noise from foreground objects.

## Program:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='LOGG', org=(50,300),
                fontFace=font, fontScale=5, color=(255,255,255),
                thickness=25, lineType=cv2.LINE_AA)
    return blank_img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

# Load original image
img = load_img()
display_img(img)

# Create kernel
kernel = np.ones((5,5), dtype=np.uint8)

## Opening
display_img(img)
white_noise = np.random.randint(low=0, high=2, size=(600,600))
white_noise = white_noise * 255
noise_img = white_noise + img
display_img(noise_img)
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
display_img(opening)

## Closing
img = load_img()
black_noise = np.random.randint(low=0, high=2, size=(600,600))
black_noise = black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img == -255] = 0
display_img(black_noise_img)
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
display_img(closing)
```
## Output:

### Display the Input Image
<img width="554" height="520" alt="Screenshot 2025-10-13 114424" src="https://github.com/user-attachments/assets/2eff9fcc-6134-4166-b1ee-40ae4225cc17" />

### Display the Noisy Image
<img width="540" height="524" alt="Screenshot 2025-10-13 114434" src="https://github.com/user-attachments/assets/0cb41129-2b83-4242-a8f7-6478cddefa80" />

### Display the Opening Image
 <img width="554" height="527" alt="Screenshot 2025-10-13 114439" src="https://github.com/user-attachments/assets/38960786-92e8-4977-9b05-4589e7a7523a" />

### Display the Black Noise Image
 <img width="552" height="531" alt="Screenshot 2025-10-13 114450" src="https://github.com/user-attachments/assets/ba559cdf-4679-4d08-8f46-98d9e575d388" />

### Display the Closing Image
 <img width="547" height="516" alt="Screenshot 2025-10-13 114456" src="https://github.com/user-attachments/assets/5ce5b7b3-59cb-4389-bbeb-513d931481c2" />

## Result
Thus, the Opening and Closing morphological operations were successfully applied on the text image using Python and OpenCV.
