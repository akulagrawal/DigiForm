import sys

import numpy as np
import cv2
import get_points

# im = cv2.imread('hqdefault.jpg')
# im = cv2.imread('sample1.jpg')
im = cv2.imread('sample4.jpg')

height, width = im.shape[:2]
im = cv2.resize(im,(int(0.3*width), int(0.3*height)), interpolation = cv2.INTER_CUBIC)
print(im.shape)

# cv2.imshow("Image", im)
# cv2.waitKey(0)

# ret,thresh = cv2.threshold(blur,180,255,0)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

fields_dict = []
while True:
    points = get_points.run(im)
    fields_dict.append(points)
    print("Points = ", points)
    temp_img = im
    cv2.rectangle(temp_img,(points[0][0],points[0][1]),(points[0][2],points[0][3]),(255,0,0),2)
    cv2.imshow("Selected", temp_img)
    key = cv2.waitKey(0)
    if key == 27:
        break

print(fields_dict)

fields = []
for val in fields_dict:
    x1 = val[0][0]
    y1 = val[0][1]
    x2 = val[0][2]
    y2 = val[0][3]

    # gray = cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # ret,thresh = cv2.threshold(blur,180,255,0)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # temp_img = thresh[x1:x2][y1:y2]     #correct
    
    temp_img = im[y1:y2, x1:x2][:]     #correct

    cv2.imshow("Selected", temp_img)
    key = cv2.waitKey(0)
    
    gray = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            x = x + x1
            y = y + y1
            # print("Ordinates:")
            # print(x, y)

            if  h>15 and w > 5 and w<30:
                l = [x, y, w, h]
                boxes.append(l)
    
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
                roi = thresh[y:y+h,x:x+w]
                # roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',temp_img)
                key = cv2.waitKey(0)
    
                if key == 27:
                    sys.exit()

    fields.append([0 + x1, 0 + y1, boxes[0][0] - x1, boxes[0][3]])
    boxes.sort(key=lambda x: x[0])
    print("Boxes = ", boxes)
    print("Fields = ", fields)
    temp_field = im[x1:boxes[0][0], y1:y1+boxes[0][3]]

    n = len(fields)
    fields1 = np.array(fields)
    fields1 = fields1.flatten()
    boxes1 = np.array(boxes)
    boxes1 = boxes1.flatten()
    
    f = open('Fields.txt','w')
    f.write(str(fields1))
    f.close()

    f=open('Boxes.txt','w')
    f.write(str(boxes1))
    f.close()
    
    # print("Field is :")
    # print(image_to_string(temp_field, lang='eng'))
# --------------------------------------------------------------------------------------------------------------------------------------------
# import sys

# import numpy as np
# import cv2
# import get_points

# # im = cv2.imread('hqdefault.jpg')
# im = cv2.imread('sample1.jpg')
# # im = cv2.imread('sample2.jpg')
# im3 = im.copy()

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# # ret,thresh = cv2.threshold(blur,180,255,0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
# # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# # fields = []
# # while True:
# #     points = get_points.run(im)
# #     fields.append(points)
# #     temp_img = im
# #     cv2.rectangle(temp_img,(points[0][0],points[0][1]),(points[0][2],points[0][3]),(0,0,255),2)
# #     cv2.imshow("Selected", temp_img)
# #     key = cv2.waitKey(0)
# #     if key == 27:
# #         break

# # print(fields)
# _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# samples =  np.empty((0,100))
# responses = []
# keys = [i for i in range(48,58)]

# X = []
# for cnt in contours:
#     if cv2.contourArea(cnt)>50:
#         [x,y,w,h] = cv2.boundingRect(cnt)

#         # if  h>30:
#         if  h>15 and h<25:
#             l = [x, y, w, h]
#             X.append(l)

#             cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
#             roi = thresh[y:y+h,x:x+w]
#             roismall = cv2.resize(roi,(10,10))
#             cv2.imshow('norm',im)
#             key = cv2.waitKey(0)

#             if key == 27:
#                 sys.exit()

# X.sort(key=lambda x: x[0])
# print(X)