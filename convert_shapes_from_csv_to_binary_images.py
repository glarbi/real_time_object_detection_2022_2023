import csv
import math
from pathlib import Path
import cv2
import numpy as np
from shapely import Polygon

BASE_DIR = Path().resolve().parent
waste = BASE_DIR / "detection_waste/shapes"
train_data = waste / "train"
test_data = waste / "test"
print('train_data :', train_data)

classes = ('other', 'plastic')

#with open("./train.csv", 'r') as file:
with open("./test.csv", 'r') as file:
    csvreader = csv.reader(file)
    image_number = 0
    for row in csvreader:
        if row:
            row = [eval(i) for i in row]
            gray_image = np.zeros((1000, 1000), dtype='uint8')
            row = [row[i:i + 2] for i in range(0, len(row), 2)]
            row = np.array(row, dtype=np.int32)
            #area = cv2.contourArea(row)
            poly = Polygon(row)
            area = poly.area
            if not(math.isclose(poly.minimum_rotated_rectangle.area, poly.area, rel_tol=0.05)):  # If the contour isn't rectangle tolerance = 5%
                #print('row: ', row)
                if area > 3000:# and area < 300000:
                    #print('area({0}): {1}'.format(image_number, area))
                    cv2.drawContours(gray_image, [row], -1, (255, 255, 255), 3)
                    #outfname = "{0}\plastic\shape_{1}.png".format(train_data, image_number)
                    outfname = "{0}\plastic\shape_{1}.png".format(test_data, image_number)
                    cv2.imwrite(outfname, gray_image)
                    image_number += 1
                    # plt.imshow(gray_image, cmap='gray')
                    # plt.title("shape_{0}.png".format(image_number))
                    # plt.show()
