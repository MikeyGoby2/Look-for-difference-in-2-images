import cv2 as cv
import numpy as np
import pyautogui as pt
import win32gui,win32con
from PIL import Image
from time import sleep
import os
from skimage.metrics import structural_similarity
import glob
import matplotlib.pyplot as plt

hwnd = win32gui.GetForegroundWindow()
win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
sleep(0.5)

# result1
pt.screenshot("imgs/screenshot.png")

img = cv.imread("imgs/screenshot.png")


cut_image1 = img[352: 721, 458: 789]
cut_image2 = img[352: 721, 824: 1155]
cv.waitKey(1)

cv.imwrite("imgs/sample1.jpg", cut_image1)
cv.imwrite("imgs/sample2.jpg", cut_image2)

image1= cv.imread("imgs/sample1.jpg")
image2= cv.imread("imgs/sample2.jpg")

difference = cv.subtract(image1, image2)
result = not np.any(difference)

cv.imwrite("results/result1.jpg", difference)
sleep(0.2)
#os.startfile("results/result1.jpg")
   
# result2 
im = Image.open('results/result1.jpg')
pixelMap = im.load()

img = Image.new( im.mode, im.size)
pixelsNew = img.load()
sleep(0.5)

for i in range(img.size[0]):
    for j in range(img.size[1]):
        if 0 in pixelMap[i,j]:
            pixelsNew[i,j] = (240,240,240)
        else:
            pixelsNew[i,j] = pixelMap[i,j]
im.close()       
img.save("results/result2.jpg") 
#img.show()

# result3
image = Image.open("results/result1.jpg")
r, g, b = image.split()
#image.show()
image = Image.merge("RGB", (b, g, r))
image.save("results/result3.jpg")
#image.show()

# results 4 to 8
before = cv.imread('imgs/sample1.jpg')
after = cv.imread('imgs/sample2.jpg')

before_gray = cv.cvtColor(before, cv.COLOR_BGR2GRAY)
after_gray = cv.cvtColor(after, cv.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(before_gray, after_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

diff = (diff * 255).astype("uint8")
diff_box = cv.merge([diff, diff, diff])

thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv.contourArea(c)
    if area > 40:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv.drawContours(mask, [c], 0, (255,255,255), -1)
        cv.drawContours(filled_after, [c], 0, (0,255,0), -1)


cv.imwrite("results/result3.jpg", before)
cv.imwrite("results/result4.jpg", after)
cv.imwrite("results/result5.jpg", diff)
cv.imwrite("results/result6.jpg", diff_box)
cv.imwrite("results/result7.jpg", mask)
cv.imwrite("results/result8.jpg", filled_after)
cv.waitKey(1)

# use glob with plt paste all togheter
file = "C:/Users/Mikey/Desktop/Python/Studio Visual Code/Gina/results/*.jpg"
glob.glob(file)
print(file)

images = [cv.imread(image) for image in glob.glob(file)]
rows = 2
cols = 4
for i in range(0, len(images), rows*cols):
    fig = plt.figure(figsize=(8,8))
    for j in range(0, rows*cols):
        fig.add_subplot(rows, cols, j+1)
        plt.imshow(images[i+j])
    plt.savefig("TotelResult.jpg")
    im_cv = cv.imread('TotelResult.jpg')
    im_rgb = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
    cv.imwrite('TotelResult2.jpg', im_rgb)
    
os.startfile("C:/Users/Mikey/Desktop/Python/Studio Visual Code/Gina/TotelResult2.jpg")
cv.waitKey(0)






