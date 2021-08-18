import torch
import torch.nn as nn
import cv2
import data_tools
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from detectors import Class_Net, Crop_Net
import timm
from torchvision import transforms as T


class NvbsCarModel():
    """Ансамбль моделей. На вход получает необработанную картинку, на выход
    отдает теги классов и километраж, если класс - panel."""
    def __init__(self, classifier, detector, ocr):
        self.classifier = classifier
        self.detector = detector
        self.ocr = ocr


    def predict(self, img):
        result = {}
        result['classes'] = self.classifier.predict(img)

        # Через детектор не проходят только картинки с классом trunk
        if 'trunk' not in result['classes']:
            detector_result = self.detector.predict(img)
            result['classes'] = result['classes'] | detector_result['classes']
            if 'panel_bbox' in detector_result:
                result['panel_bbox'] = detector_result['panel_bbox']

        # Через OCR проходят только картинки с классом panel
        if 'panel_bbox' in result:
            cropped_image = data_tools.crop(img, result['panel_bbox'])
            result['mileage'] = self.ocr.predict(cropped_image)

        return result


class CarsClassifier():
    """Классификатор, отдает теги классов."""
    def __init__(self, path, threshold=0.5):
        self.model = EfficientNet.from_name('efficientnet-b4')
        self.model._fc = torch.nn.Linear(in_features=1792, out_features=13, bias=True)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.threshold = threshold
        self.size = 224
        self.transforms = A.Compose(
            [
                A.Normalize(
                    mean=[0.4595, 0.4634, 0.4740],
                    std=[0.2418, 0.2418, 0.2428],
                ),
                ToTensorV2(),
            ]
        )
        self.model.eval()

    def predict(self, img):
        X = self._preprocess(img)
        preds = self.model(X)
        preds = torch.sigmoid(preds)
        preds = preds.cpu().detach().numpy().reshape(-1)
        labels = np.where(preds > self.threshold)[0]
        if len(labels) > 0:
            labels = set([data_tools.label_to_classes[x] for x in labels])
        else:
            labels = {}
        return labels

    def _preprocess(self, img):
        X = img.astype(float)
        X = cv2.resize(X, (self.size, self.size))
        X = self.transforms(image=X)["image"]
        X = X.reshape(1, 3, self.size, self.size)
        return X


class CarsDetector():
    """Детектор/классификатор на повреждения кузова, цифровую панель, мусор и
    грязь в салоне. Отдает теги классов и координаты ббокса, если панель.
    """
    classes = {
        '0': 'damaged',
        '1': 'panel',
        '2': 'rubbish',
        '3': 'dirt'
    }

    def __init__(self, path):
        weights = torch.load(path)['model'].state_dict()
        new_weights = {}
        for key in weights.keys():
            new_key = 'model.' + key
            new_weights[new_key] = weights[key]

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', classes=4)
        self.model.load_state_dict(new_weights)

    def predict(self, img):
        # Считаем что грузим через cv2, поэтому конверт BGR -> RGB
        img = img[:, :, ::-1]
        pred = self.model(img)
        table = pred.pandas().xyxy[0]
        table['name'] = table['name'].replace(self.classes)

        result = {
            'classes': set(table['name'].unique())
            }

        if 'panel' in result['classes']:
            subset = table[table['name'] == 'panel'].iloc[0]
            result['panel_bbox'] = [
                subset['xmin'],
                subset['ymin'],
                subset['xmax'],
                subset['ymax']
                ]
        return result

class PanelDigitsDetector():
    """Детектор/классификатор на цифры одометра на панели автомобиля.
    Отдает показания одометра.
    """


    def __init__(self, path_to_class, path_to_crop):
        model_backbone_1 = timm.create_model('{}'.format('resnet18d'), pretrained = False)
        model_backbone_2 = timm.create_model('{}'.format('resnet18d'), pretrained = False)

        self.crop_model= Crop_Net(model_backbone_1)
        self.crop_model.load_state_dict(torch.load(path_to_crop, map_location=torch.device('cpu'))['state_dict'])
        self.crop_model.eval()

        self.class_model = Class_Net(model_backbone_2)
        self.class_model.load_state_dict(torch.load(path_to_class, map_location=torch.device('cpu'))['state_dict'])
        self.class_model.eval()

    def predict(self, img):

        trans_crop = T.Compose([
            T.Resize((25, 25))])

        img = self._preprocess(img)
        out_crop = self.crop_model(img).squeeze(1).detach().numpy()*256

        number = []

        for box in out_crop[::-1]:
            if ((box < 0).sum() > 0) or (box[2]+box[3] < 10):
                None
            else:
                box = [round(x) for x in box]
                img_c = img.squeeze(0)[:, box[1]:box[1]+box[2], box[0]:box[0]+box[3]]
                img_c = trans_crop(img_c)
                digit = self.class_model(img_c.unsqueeze(0))
                digit = torch.argmax(digit, 1)
                number.append(digit.numpy()[0])

        number = self.magic(number[::-1])

        return number

    def _preprocess(self, img):
        trans_class = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256))])

        img = trans_class(img).unsqueeze(0)

        return img

    def magic(self, numList):
        'makes int out of list'
        s = ''.join(map(str, numList))
        return int(s)

