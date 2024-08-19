import cv2
import numpy as np


image = cv2.imread('self2222.jpg')
image = cv2.resize(image,(400,400))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sigma = 1.4
# 1. اعمال فیلتر گوسی برای کاهش نویز
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 1.4)

# 2. اعمال فیلتر میانه
#ابتدا فقط فیلتر گوسی اعمال شده است اما به دلیل تشخیص لبه های 
#اشتباه زیادی  از فیلتر میانه برای دقت بیشرت و کاهش نویز استفاده شده است
median_image = cv2.medianBlur(blurred_image, 5)

low_threshold = 50
high_threshold = 150
# 3. استخراج لبه‌ها با استفاده از Canny
edges = cv2.Canny(median_image, 50, 150)

# 4. استفاده از تبدیل هاف دایره برای تشخیص عنبیه
circles = cv2.HoughCircles(edges,
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
# cv2.imwrite('edge_canny.jpg',edges)
# cv2.imwrite('edge_canny_high.jpg',edges)
# cv2.imwrite('eye_detect_canny.jpg',image)
cv2.imshow('output', image)
#cv2.imshow('outpuut', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
