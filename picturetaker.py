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
url = f"rtsp://{IP}/mpeg4/"
cap = cv.VideoCapture(url)
i = 0
while True:
    ret, frame = cap.read()
    if frame is not None:
        frame_rezise = cv.resize (frame, (1024, 768))
        cv.imshow('Video Stream', frame_rezise)
        
        
    # Press C on keyboard to save Image
    if cv.waitKey(1) & 0xFF == ord('c'):
        cv.imwrite('FOD_4m' + str(i) + '_.jpg', frame)
        i += 1
        
    # Press Q on keyboard to  exit
    if cv.waitKey(1) & 0xFF == ord('q'):
      cap.release()
      break