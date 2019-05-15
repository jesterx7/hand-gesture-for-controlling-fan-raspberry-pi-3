from sklearn.metrics import pairwise
import numpy as np
import cv2
import imutils

class GestureDetector:
    def __init__(self):
        pass
    
    def detect(self, thresh, cnt):
        #menggunakan metode convex hull terhadap contour
        hull = cv2.convexHull(cnt)
        #mengambil 4 titik paling kiri, kanan, atas, bawah
        extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
        extRight = tuple(hull[hull[:, :, 0].argmax()][0])
        extTop = tuple(hull[hull[:, :, 1].argmin()][0])
        extBot = tuple(hull[hull[:, :, 1].argmax()][0])
        
        #mencari centroid tangan
        cX = (extLeft[0] + extRight[0]) // 2
        cY = (extTop[1] + extBot[1]) // 2
        cY += (cY * 0.15)
        cY = int(cY)
        
        #menghitung euclidean distance dari keempat titik tadi
        D = pairwise.euclidean_distances([(cX, cY)], Y=[extLeft, 
                                          extRight, extTop, extBot])[0]
        #mengambil titik terjauh
        maxDist = D[D.argmax()]
        #menentukan jari-jari lingkaran dari jarak terjauh
        r = int(0.7 * maxDist)
        #meghitung keliling lingkaran
        circum = 2 * np.pi * r
        
        #membuat array gambar dengan value 0 (hitam)
        circleROI = np.zeros(thresh.shape[:2], dtype="uint8")
        #menggambar lingkaran pada circleROI
        cv2.circle(circleROI, (cX, cY), r, 255, 1)
        #mencari perpotongan tangan dengan lingkaran yang sudah dibuat
        circleROI = cv2.bitwise_and(thresh, thresh, mask=circleROI)
        #mencari contour dari hasil perpotongan tersebut
        cnts = cv2.findContours(circleROI.copy(), cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_NONE)
        #mengambil contour dengan imutils grab contours
        cnts = imutils.grab_contours(cnts)
        #variabel untuk menyimpan jumlah jari
        total = 0
        
        #memberi bounding rect disetiap contour
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            #rules untuk mengenali jari
            if c.shape[0] < circum * 0.25 and (y+h) < cY + (cY * 0.25):
                total += 1
        
        return total
    
    #menggambar text pada frame
    @staticmethod
    def drawText(roi, i, val, color=(0, 255, 0)):
        cv2.putText(roi, str(val), ((i * 50) + 20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    #menggambar bounding box
    @staticmethod
    def drawBox(roi, i, color=(0, 0, 255)):
        cv2.rectangle(roi, ((i * 50) + 10, 10), 
                      ((i * 50) + 50, 60), color, 2)