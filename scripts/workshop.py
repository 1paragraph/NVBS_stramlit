import nvbs_models
import cv2
from cars_parser import CarsParser
import os
import streamlit as st
import numpy as np

m = nvbs_models.NvbsCarModel(
    classifier=nvbs_models.CarsClassifier('./model_weights/b4.pt'),
    detector=nvbs_models.CarsDetector('./model_weights/yolov5m.pt'),
    ocr=nvbs_models.PanelDigitsDetector('./model_weights/class_resnet18d.pt',
                                        './model_weights/bbox_model_resnet18d.pt')
    )

uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    st.write(opencv_image)

st.write(m(opencv_image))

# cp = CarsParser(m)
# path = './test_data'
# cp.parse(path)
# result = cp.get_report()
# result.to_csv('result.csv', index=False)
# print(result)