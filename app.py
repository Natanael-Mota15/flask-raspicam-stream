from flask import Flask, render_template, Response
from rpi_camera import RPiCamera
from imageai.Detection import ObjectDetection
import cv2

cam_feed=cv2.VideoCapture(0)
obj_detect = ObjectDetection()
obj_detect.setModelTypeAsYOLOv3
obj_detect.setModelPath(r"C:/Datasets/yolo.h5")




app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


#the generator, a special type of function that yields, instead of returns.
def gen(camera):

    while True:
        ret, img = cam_feed.read()   
        annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                    input_type="array",
                      output_type="array",
                      display_percentage_probability=False,
                      display_object_name=True)

        cv2.imshow("", annotated_image)     
    
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break


        # Each frame is set as a jpg content type. Frame data is in bytes.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/stream')
def stream():
    
    feed = Response(gen(RPiCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

    print(type(feed))
    return feed


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True )