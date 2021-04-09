import cv2


class IpCam:
    def __init__(self, url):
        self.url = "https://" + url + ":8080/video"

    def get_cam(self):
        return cv2.VideoCapture(self.url)
