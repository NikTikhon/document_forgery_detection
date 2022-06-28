import datetime
import numpy as np
import torch
import cnn
import cv2 as cv
from torchvision import transforms
import argparse
from flask import Flask
import math
from detect_text import get_masks
from utils import drow_mask, get_patches, bbox_patch, get_prediction, fix_bbox


app = Flask(__name__)

MODE = 'DEBUG'
# MODE = 'RELEASE'
CREATE_MAP = 1
THRESHOLD = 0.02
MAPS_PATH = 'map/'
MODEL_PATH = 'models/model_weights.pth'


@app.route('/detect/<path:image_path>')
def detect(image_path):
    td = TamperingDetection(THRESHOLD)
    image = cv.imread(image_path)
    (class_id, result, map) = td.detect(image_path, image)
    if CREATE_MAP == 1:
        return {'class': class_id, 'result': result, 'map': map}
    else:
        return {'class': class_id, 'result': result}


class TamperingDetection:
    def __init__(self, threshold):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.cnn = cnn.Net().to(self.device)
            self.cnn.load_state_dict(torch.load(MODEL_PATH,
                                               map_location=lambda storage, loc: storage))
            self.cnn.eval()
            self.threshold = threshold


    def detect(self, image_path, image:np.ndarray):
        image_map = image.copy()
        stride = 8
        patches = get_patches(image, stride=stride)
        transform = transforms.Compose([transforms.ToTensor()])
        predictions = []
        for patch in patches:
            patch_image = patch[0]
            tensor = transform(patch_image)
            tensor = tensor.to(self.device)
            tensor = tensor.unsqueeze_(0)
            prediction = get_prediction(self.cnn, tensor)
            if prediction[0, 0] > 0.92:
                prediction = 0
            else:
                prediction = 1
            # prediction = prediction.argmax(1).item()
            # if CREATE_MAP == 1 and prediction == 0:
            #     image_map = bbox_patch(image_map, patch, stride)
            predictions.append(prediction)
        predictions = np.array(predictions)
        count_fake_pred = len(predictions[predictions == 0])
        # prop_fake = count_fake_pred / len(patches)
        prop_fake = count_fake_pred
        score = 1 / (1 + math.exp(-((prop_fake / THRESHOLD) - 2.5)))
        if not prop_fake >= self.threshold:
            return (1, image_path + ' - RESULT: Original. SCORE: {:10.2f}'.format(1 - score), image_map)
        else:
            return (0, image_path + ' - RESULT: Tampered. SCORE: {:10.2f}'.format(score), image_map)


    def detect_masks(self, image_path, image:np.ndarray, bboxes: list):
        image_map = image.copy()
        masks = np.zeros_like(image_map)
        cnt_fake_all_image = 0
        count_patches = 0
        stride = 16
        for bbox in bboxes:
            bbox_fix = fix_bbox(bbox)
            image_mask = image[bbox_fix[0]:bbox_fix[1], bbox_fix[2]: bbox_fix[3]]
            patches = get_patches(image_mask, stride=stride)
            transform = transforms.Compose([transforms.ToTensor()])
            predictions = []
            for patch in patches:
                patch_image = patch[0]
                tensor = transform(patch_image)
                tensor = tensor.to(self.device)
                tensor = tensor.unsqueeze_(0)
                prediction = get_prediction(self.cnn, tensor)
                if prediction[0, 0] > 0.90:
                    prediction = 0
                else:
                    prediction = 1
                predictions.append(prediction)
            predictions = np.array(predictions)
            count_fake_pred = len(predictions[predictions == 0])
            # prop_fake = count_fake_pred
            cnt_fake_all_image += count_fake_pred
            count_patches += len(patches)
            threshold_exceeded = count_fake_pred >= self.threshold
            masks = drow_mask(masks, bbox, threshold_exceeded)
        image_map = cv.addWeighted(image_map, 0.9, masks, 1, 0.0)
        prop_fake = cnt_fake_all_image / len(bboxes)
        score = 1 / (1 + math.exp(-((prop_fake / self.threshold) - 1)))
        if not prop_fake >= self.threshold:
            return (1, image_path + ' - RESULT: Original. SCORE: {:10.2f}'.format(1 - score), image_map)
        else:
            return (0, image_path + ' - RESULT: Tampered. SCORE: {:10.2f}'.format(score), image_map)


    def set_thresh(self, thr):
        self.threshold = thr


if __name__ == '__main__':
    if MODE == 'DEBUG':
        main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
        main_arg_parser.add_argument("--image-path", type=str, required=True,
                                     help="path to content image")
        main_arg_parser.add_argument("--model-weight", type=str, required=False,
                                     help="saved model to be used for detection fake image.")
        main_arg_parser.add_argument("--create-map", type=int, required=True)
        main_arg_parser.add_argument("--out-map-path", type=str, required=False)
        main_arg_parser.add_argument("--threshold", type=float, required=True)
        main_arg_parser.add_argument("--is-full-image", type=int, required=True)
        args = main_arg_parser.parse_args()
        if args.model_weight is not None:
            MODEL_PATH = args.model_weight
        CREATE_MAP = args.create_map
        THRESHOLD = args.threshold
        if not args.is_full_image:
            result = detect(args.image_path)
            print(str(result['class']) + result['result'])
            if args.create_map == 1:
                image_map = result['map']
                if args.out_map_path is not None:
                    name_file = (args.image_path).split('\\')[-1].split('.')[0] + f'_map.png'
                    cv.imwrite(args.out_map_path + '\\' + name_file, image_map)
                else:
                    now = datetime.datetime.now()
                    cv.imwrite(f'maps/{now.hour}_{now.minute}_{now.second}.png', image_map)
        else:
            td = TamperingDetection(args.threshold)
            name_file = (args.image_path).split('\\')[-1]
            masks = get_masks(args.image_path)
            image = cv.imread(args.image_path)
            (class_id, result, map) = td.detect_masks(args.image_path, image, masks)
            print(result)
            cv.imwrite(args.out_map_path + '\\' + name_file, map)


    elif MODE == 'RELEASE':
        app.run()

