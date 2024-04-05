import cv2 as cv

IP = "192.168.16.1" #Replace with current IP


urls = [
    "rtsp://" + IP + "/avc/ch1",
    "rtsp://" + IP + "/mjpg/ch1",
    "rtsp://" + IP + "/mpeg4/ch1",
    f"rtsp://{IP}/mpeg4/",
    "rtsp://" + IP + "/avc/",
    "rtsp://" + IP + "/mjpg/",
    0
]
url = "192.168.16.1" + f"rtsp://{IP}/mpeg4/"
cap = cv.VideoCapture(url)
i = 0
while True:
    cap, frame = cv.read()
    
    # Press C on keyboard to save Image
    if cv.waitKey(0) & 0xFF == ord('c'):
        cv.imwrite(str(i) + '_screwdriver_frame.jpg', frame)
        i += 1
        
    # Press Q on keyboard to  exit
    if cv.watiKey(0) & 0xFF == ord('q'):
      cap.release()
      break