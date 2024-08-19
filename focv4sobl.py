import cv2
import numpy as np

image = cv2.imread('self2222.jpg')
image = cv2.resize(image,(400,400))
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#اعمال فلیتر سوبل
sobel_x = cv2.Sobel(gray_image,cv2.CV_64F,1,0)
sobel_y = cv2.Sobel(gray_image,cv2.CV_64F,0,1)

#محاسبه اندازه گرادیان یا تصویر گرادیان
m = np.sqrt(sobel_x**2 + sobel_y**2)

# تبدیل مقادیر به 8 بیت
m_unit8 =  m.astype(np.uint8)



cv2.imwrite('edge_sobel.jpg',m_unit8)
cv2.imshow('output',m_unit8)
cv2.waitKey(0)