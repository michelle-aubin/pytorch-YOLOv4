import cv2

data_path = '../BBOX-LABELS-608/'


with open(data_path+'train.txt') as f:
    for line in f:
        data = line.strip().split(" ")
        img_path = data[0]
        img = cv2.imread(img_path)
        for box in data[1:]:
            coords = box.split(",")
            cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255,0,0), 2)
        cv2.imshow("boxes", img)
        cv2.waitKey(0)
