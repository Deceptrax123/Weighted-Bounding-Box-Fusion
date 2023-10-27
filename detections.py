import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


# draw the outputs using opencv tools
def draw_bbox_conf(image, boxes, scores, color=(255, 0, 0), thickness=-1):
    overlay = image.copy()

    font_size = 0.25+0.07*min(overlay.shape[:2])/100
    font_size = max(font_size, 0.5)
    font_size = min(font_size, 0.8)
    text_offset = 7

    for box, score in zip(boxes, scores):
        xmin = box[0]
        ymin = box[1]
        xmax = box[0]+box[2]
        ymax = box[1]+box[3]

        overlay = cv2.rectangle(overlay, (xmin, ymin),
                                (xmax, ymax), color, thickness)
        display_text = f"wheat_head: {score[0]:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)

        cv2.rectangle(overlay, (xmin, ymin), (xmin+text_width+text_offset,
                      ymin-text_height-int(15*font_size)), color, thickness=-1)

        overlay = cv2.putText(overlay, display_text, (xmin+text_offset, ymin-int(10*font_size)),
                              cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)


def detect():
    res = pd.read_csv("./Results/test_inpaint.csv",
                      converters={'boxes': pd.eval, 'scores': pd.eval})

    img_test = '/Volumes/T7 Shield/Smudge/Datasets/Wheat_Head_detection/test/0ade56ba5d47c384378fc185efe9f12e4143664c1dbb45f34473bdbcc67daaf8.png'
    id = "0ade56ba5d47c384378fc185efe9f12e4143664c1dbb45f34473bdbcc67daaf8.png"

    boxes = np.array(res.loc[res['image_id'] == id]['boxes'])[0]
    scores = np.array(res.loc[res['image_id'] == id]['scores'])[0]

    cv_img = cv2.imread(img_test)
    image_boxed = draw_bbox_conf(
        cv_img, boxes, scores)

    cv2.imshow('experiment', image_boxed)
    cv2.waitKey(0)


def main():
    detect()


if __name__ == '__main__':
    main()
