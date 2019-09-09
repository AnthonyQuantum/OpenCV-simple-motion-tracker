import cv2

OUTPUT_VIDEO_RESOLUTION = (1280, 720)

MIN_CONTOUR_AREA = 900
AREA_SMALL_UBOUND = 10000
AREA_MEDIUM_UBOUND = 25000

COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_BLACK = (0,0,0)

video = cv2.VideoCapture(0)

frame1_read, frame1 = video.read()
frame2_read, frame2 = video.read()
while video.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = list(filter(lambda contour: cv2.contourArea(contour) > MIN_CONTOUR_AREA, contours))
    cv2.drawContours(frame1, contours, -1, COLOR_BLUE, 1)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)
        if area < AREA_SMALL_UBOUND:
            rect_color = COLOR_GREEN
        elif area < AREA_MEDIUM_UBOUND:
            rect_color = COLOR_YELLOW
        else:
            rect_color = COLOR_RED
        
        cv2.rectangle(frame1, (x,y), (x+w,y+h), rect_color, 2)
    
    cv2.putText(frame1, 'Moving objects: {}'.format(len(contours)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_BLACK, 2)
    
    image = cv2.resize(frame1, OUTPUT_VIDEO_RESOLUTION)
    cv2.imshow('Motion tracker', image)

    frame1 = frame2
    frame2_read, frame2 = video.read()

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()