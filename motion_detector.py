#Import library yang dibutuhkan
import cv2
import imutils

class MotionDetector:
	#Init
	def __init__(self):
		#Background
		self.bg = None

	def update(self, image):
		#update gambar background
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

	def detect(self, image, tVal=25):
		#menyimpan perbedaan gambar /deteksi gerakan
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		#thresholding gambar
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
		#mencari contour
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
		#ubah contour dengan imutils
		cnts = imutils.grab_contours(cnts)

		#tidak ada pergerakan
		if len(cnts) == 0:
			return None

		#return value
		return (thresh, max(cnts, key=cv2.contourArea))