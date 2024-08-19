import cv2
import numpy as np


image = cv2.imread('self2222.jpg')
image = cv2.resize(image,(400,400))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sigma = 1.4
# 1. اعمال فیلتر گوسی برای کاهش نویز
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

# 2. اعمال فیلتر میانه
#ابتدا فقط فیلتر گوسی اعمال شده است اما به دلیل تشخیص لبه های 
#اشتباه زیادی  از فیلتر میانه برای دقت بیشرت و کاهش نویز استفاده شده است
median_image = cv2.medianBlur(blurred_image,5)

# 3. محاسبه لاپلاسین
laplacian = cv2.Laplacian(median_image, cv2.CV_64F)

# 4. یافتن نقاط صفری
threshold = 0.1
abs_laplacian = np.absolute(laplacian)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(abs_laplacian)
_, zero_crossings = cv2.threshold(abs_laplacian, min_val + (max_val - min_val) * 0.2, 255, cv2.THRESH_BINARY)
zero_crossings = np.uint8(zero_crossings)

# 5. استفاده از تبدیل هاف دایره برای تشخیص عنبیه
circles = cv2.HoughCircles(zero_crossings,
    cv2.HOUGH_GRADIENT,
    dp=1.2,           
    minDist=100,      
    param1=50,       
    param2=30,        
    minRadius=20,     
    maxRadius=60)     

# بررسی و ترسیم دایره‌های شناسایی شده
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # ترسیم دایره
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # ترسیم مرکز دایره
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# نمایش تصویر خروجی
# cv2.imwrite('edge_marr_hildreth.jpg',zero_crossings)
# cv2.imwrite('edge_marr_hildreth_high.jpg',zero_crossings)
# cv2.imwrite('eye_detect_marr_hildreth.jpg',image)
cv2.imshow('output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
