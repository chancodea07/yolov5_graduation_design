# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
from yolov5 import Darknet
from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_boxes, check_imshow
from utils.dataloaders import letterbox
from flask import Response, request, send_file
from flask import Flask
from flask import render_template
from flask_cors import CORS, cross_origin
from PIL import Image
import time
import torch
import json
import cv2
import io

# initialize a flask object
app = Flask(__name__)
CORS(app)

# initialize the video stream and allow the camera sensor to warmup
with open('yolov5_config.json', 'r', encoding='utf8') as fp:
    opt = json.load(fp)
    print('[INFO] YOLOv5 Config:', opt)

darknet = Darknet(opt)
if darknet.webcam:
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(darknet.source,
                          img_size=opt["imgsz"],
                          stride=darknet.stride)
else:
    dataset = LoadImages(darknet.source,
                         img_size=opt["imgsz"],
                         stride=darknet.stride)
time.sleep(2.0)


# view_img = True
@app.route("/")
def index():
    # return the rendered template
    return render_template("greeting.html")


@app.route('/video')
def video():
    return render_template('video.html')


def detect_gen(dataset, feed_type):
    view_img = check_imshow()
    t0 = time.time()
    for path, img, img0s, vid_cap in dataset:
        img = darknet.preprocess(img)

        t1 = time.time()
        pred = darknet.model(img, augment=darknet.opt["augment"])[0]  # 0.22s
        pred = pred.float()
        pred = non_max_suppression(pred, darknet.opt["conf_thres"],
                                   darknet.opt["iou_thres"])
        t2 = time.time()

        pred_boxes = []
        for i, det in enumerate(pred):
            if darknet.webcam:  # batch_size >= 1
                feed_type_curr, p, s, im0, frame = "Camera_%s" % str(
                    i), path[i], '%g: ' % i, img0s[i].copy(), dataset.count
            else:
                feed_type_curr, p, s, im0, frame = "Camera", path, '', img0s, getattr(
                    dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4],
                                         im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {darknet.names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls_id in det:
                    lbl = darknet.names[int(cls_id)]
                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    score = round(conf.tolist(), 3)
                    label = "{}: {}".format(lbl, score)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(
                        xyxy[2]), int(xyxy[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, score))
                    if view_img:
                        darknet.plot_one_box(
                            xyxy,
                            im0,
                            color=(255, 0,
                                   0) if label.split(':')[0] == 'masking' else
                            (0, 0, 255),
                            label=label)
            # Print time (inference + NMS)
            # print(pred_boxes)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            if feed_type_curr == feed_type:
                frame = cv2.imencode('.jpg', im0)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'Camera_1':
        return Response(detect_gen(dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def draw_box_corner(draw_img, bbox, length, corner_color):
    # Top Left
    cv2.line(draw_img, (bbox[0], bbox[1]), (bbox[0] + length, bbox[1]),
             corner_color,
             thickness=3)
    cv2.line(draw_img, (bbox[0], bbox[1]), (bbox[0], bbox[1] + length),
             corner_color,
             thickness=3)
    # Top Right
    cv2.line(draw_img, (bbox[2], bbox[1]), (bbox[2] - length, bbox[1]),
             corner_color,
             thickness=3)
    cv2.line(draw_img, (bbox[2], bbox[1]), (bbox[2], bbox[1] + length),
             corner_color,
             thickness=3)
    # Bottom Left
    cv2.line(draw_img, (bbox[0], bbox[3]), (bbox[0] + length, bbox[3]),
             corner_color,
             thickness=3)
    cv2.line(draw_img, (bbox[0], bbox[3]), (bbox[0], bbox[3] - length),
             corner_color,
             thickness=3)
    # Bottom Right
    cv2.line(draw_img, (bbox[2], bbox[3]), (bbox[2] - length, bbox[3]),
             corner_color,
             thickness=3)
    cv2.line(draw_img, (bbox[2], bbox[3]), (bbox[2], bbox[3] - length),
             corner_color,
             thickness=3)


def draw_label_type(draw_img, bbox, label_color):
    label = bbox[-2] + ' ' + str(bbox[-1])
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img, (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1)
        cv2.putText(draw_img,
                    label, (bbox[0], bbox[1] + labelSize + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0),
                    thickness=1)
    else:
        cv2.rectangle(draw_img, (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1)
        cv2.putText(draw_img,
                    label, (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0),
                    thickness=1)


@app.route('/pic')
def pic_page():
    return render_template('pic.html')


@app.route('/pic_detection', methods=['POST'])
def pic_detect():
    from werkzeug.utils import secure_filename
    if request.method == 'POST':
        f = request.files['image']
        picpath = 'static/' + secure_filename(f.filename)
        f.save(picpath)
        # 读取配置文件，另外建立模型
        with open('yolov5_config.json', 'r', encoding='utf8') as fp:
            opt = json.load(fp)
        source = opt['source'] = picpath
        dnet = Darknet(opt)
        dataset = LoadImages(source, img_size=opt["imgsz"], stride=dnet.stride)
        # 得到预测坐标，由于是API形式，所以默认不显示图片
        arr = dnet.detect(dataset=dataset, view_img=False)
        img = Image.open(f)  # 将BytesIO对象转换为PIL图片
        img_array = np.array(img)  # 将PIL图片转换为numpy数组
        # 将RGB颜色空间转换为BGR颜色空间（OpenCV默认使用BGR）
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        box_color = (0, 0, 255)
        # 多目标识别绘图
        for item in arr:
            box_color_mask = (0, 0, 255)
            box_color_unmask = (255, 0, 0)
            # 不同类别不同框
            if item[4] == 'masking':
                box_color = box_color_mask
            else:
                box_color = box_color_unmask
            cv2.rectangle(img_bgr, (item[0], item[1]), (item[2], item[3]),
                          color=box_color,
                          thickness=2)
            draw_label_type(img_bgr, item, box_color)
            draw_box_corner(img_bgr, item, 10, (0, 255, 128))
        _, img_encoded = cv2.imencode('.jpg', img_bgr)
        img_bytes = io.BytesIO(img_encoded)  # 将二进制数据转换为BytesIO对象
        return send_file(img_bytes, mimetype='image/jpeg')  # 返回BytesIO对象给客户端


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5000", threaded=True)
