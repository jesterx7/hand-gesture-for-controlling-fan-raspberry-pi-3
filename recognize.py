from gesture_detector import GestureDetector
from motion_detector import MotionDetector
import RPi.GPIO as GPIO
import numpy as np
import imutils
import time
import cv2

#deklarasi kamera
camera = cv2.VideoCapture(0)
#deklarasi ROI (Region of Interest)
(top, right, bot, left) = np.int32(("100, 300, 375, 580").split(","))
#deklarasi class yang sudah dibuat
gd = GestureDetector()
md = MotionDetector()
#deklarasi pin yang dipakai
fan_pin = 12
led_pin = [16, 18, 22]
buzzer_pin = 32
#setup GPIO sesuai PIN
GPIO.setmode(GPIO.BOARD)
GPIO.setup(fan_pin, GPIO.OUT)
GPIO.setup(led_pin[0], GPIO.OUT)
GPIO.setup(led_pin[1], GPIO.OUT)
GPIO.setup(led_pin[2], GPIO.OUT)
GPIO.setup(buzzer_pin, GPIO.OUT)
#setting pwm PIN (fan dan buzzer)
pwm = GPIO.PWM(fan_pin, 100)
buzzer_pwm = GPIO.PWM(buzzer_pin, 100)
#menjalankan fan dan buzzer dengan voltase 0
pwm.start(0)
buzzer_pwm.start(0)
#setting frekuensi buzzer menjadi 500
buzzer_pwm.ChangeFrequency(500)

#deklarasi variabel yang dibutuhkan
numFrames = 0 #menghitung frame yang diolah
gesture = None #menampung pengenalan jumlah jari
values = None #menampung hasil jari
img_finger = None #menampung hasil jari untuk digambarkan pada frame

while True:
    #membaca frame pada kamera
    (grabbed, frame) = camera.read()
    #resize frame
    frame = imutils.resize(frame, width=600)
    #flip frame agar terlihat seperti cermin
    frame = cv2.flip(frame, 2)
    #copy frame ke variabel clone
    clone = frame.copy()
    #mengambil tinggi dan lebar dari frame
    (frameH, frameW) = frame.shape[:2]
    
    #crop frame pada ROI untuk dideteksi
    roi = frame[top:bot, right:left]
    #dijadikan ke grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #smoothing gambar dengan gaussian blur
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    
    #frame pertama digunakan sebagai background
    if numFrames < 1:
        md.update(gray)
    else :
        #deteksi gerakan
        skin = md.detect(gray)
        #jika ada gerakan
        if skin is not None:
            (thresh, c) = skin
            #gambar contour
            cv2.drawContours(clone, [c + (right,top)], -1, (0, 255, 0), 2) #draw objek skin dengan menambahkan nilai kontur ke batasan box
            #pengenalan dan menghitung jumlah jari terangkat
            fingers = gd.detect(thresh, c)

            if gesture is None:
                gesture = [1, fingers]
            else:
                #jika gesture / jumlah jari sama dengan sebelumnya
                if gesture[1] == fingers:
                    gesture[0] +=1
                    #jika jumlah jari sama selama 25 frame
                    if gesture[0] >= 25:
                        #maka disimpulkan bahwa jumlah jari tersebut benar terhitung
                        values = fingers
                        print(values)
                        gesture = None
                else:
                    gesture = None
    #jika ada tangan yang terdeteksi maka hasilnya akan digambar di frame
    if img_finger is not None:
        GestureDetector.drawBox(clone, 0)
        GestureDetector.drawText(clone, 0, img_finger)
    
    #rules untuk mengontrol kipas, led, dan buzzer jika ada tangan terdeteksi
    if values is not None:
        img_finger = values
        if values == 1:
            #buzzer berbunyi
            buzzer_pwm.ChangeDutyCycle(99)
            #loops untuk timer buzzer berbunyi
            for n in range(20, 0, -5):
                time.sleep(.15)
            #buzzer selesai berbunyi
            buzzer_pwm.ChangeDutyCycle(0)
            #mengatur kecepatan kipas angin
            pwm.ChangeDutyCycle(100)
            time.sleep(.2)
            pwm.ChangeDutyCycle(30)
            #untuk mengatur lampu yang hidup (1 lampu hidup 2 mati)
            GPIO.output(led_pin[0], True)
            GPIO.output(led_pin[1], False)
            GPIO.output(led_pin[2], False)
            values = None
        elif values == 2:
            buzzer_pwm.ChangeDutyCycle(99)
            for n in range(20, 0, -5):
                time.sleep(.15)
            buzzer_pwm.ChangeDutyCycle(0)
            GPIO.output(led_pin[0], True)
            GPIO.output(led_pin[1], True)
            GPIO.output(led_pin[2], False)
            pwm.ChangeDutyCycle(100)
            pwm.ChangeDutyCycle(60)         
            values = None
        elif values > 2:
            buzzer_pwm.ChangeDutyCycle(99)
            for n in range(20, 0, -5):
                time.sleep(.15)
            buzzer_pwm.ChangeDutyCycle(0)
            pwm.ChangeDutyCycle(100)
            GPIO.output(led_pin[0], True)
            GPIO.output(led_pin[1], True)
            GPIO.output(led_pin[2], True)
            values = None
        else:
            buzzer_pwm.ChangeDutyCycle(99)
            for n in range(20, 0, -5):
                time.sleep(.15)
            buzzer_pwm.ChangeDutyCycle(0)
            GPIO.output(led_pin[0], False)
            GPIO.output(led_pin[1], False)
            GPIO.output(led_pin[2], False)
            pwm.ChangeDutyCycle(0)
            values = None
    
    #menggambar ROI pada frame
    cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
    numFrames +=1
    
    #menampilkan gambar yang diambil
    cv2.imshow("Frame", clone)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
#cleaning (mematikan seluruh input / output, camera, dan window yang menyala) 
GPIO.output(led_pin[0], False)
GPIO.output(led_pin[1], False)
GPIO.output(led_pin[2], False)
buzzer_pwm.stop()
pwm.stop()
GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()