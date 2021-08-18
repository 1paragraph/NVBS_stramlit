import nvbs_models
import cv2
from cars_parser import CarsParser

m = nvbs_models.NvbsCarModel(
    classifier=nvbs_models.CarsClassifier('./model_weights/b4.pt'),
    detector=nvbs_models.CarsDetector('./model_weights/yolov5m.pt'),
    ocr=nvbs_models.PanelDigitsDetector('./model_weights/class_resnet18d.pt',
                                        './model_weights/bbox_model_resnet18d.pt')
    )

cp = CarsParser(m)
path = '../test_data'
cp.parse(path)
result = cp.get_report()
result.to_csv('result.csv', index=False)
print(result)