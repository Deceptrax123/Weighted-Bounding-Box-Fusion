import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


def convert_to_crowd():
    load_dotenv('./.env')

    res_path = os.getenv("RESULTS_CSV")
    df = pd.read_csv(res_path, converters={
        "boxes": pd.eval, "scores": pd.eval
    })

    box_formatted = list()
    ids = list()

    oids = list(df['image_id'])
    boxes = list(df['boxes'])

    dict_boxes = dict(zip(oids, boxes))

    for i in dict_boxes.keys():
        image_name = i.split(".png")[0]
        ids.append(image_name)

        prediction = dict_boxes[i]
        if prediction == []:
            box_formatted.append('no_box')
        else:
            formatted = ""
            for k, i in enumerate(prediction):
                a, b, c, d = str(i[0]), str(i[1]), str(i[2]), str(i[3])

                if k != len(prediction)-1:
                    pred = a+' '+b+' '+c+' '+d+';'
                    formatted += pred
                else:
                    pred = a+' '+b+' '+c+' '+d
                    formatted += pred
            box_formatted.append(formatted)

    df = pd.DataFrame({'image_name': ids, 'PredString': box_formatted})

    crowd_save_path = os.getenv("AI_CROWD_SAVE")
    df.to_csv(crowd_save_path)


def main():
    convert_to_crowd()


if __name__ == '__main__':
    main()
