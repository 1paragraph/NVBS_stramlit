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

st.title('НВБС')
st.subheader('Инструмент для определения состояний автомобилей и показаний одометра')


st.text('Загрузите картинку с автомобилем сюда в формате .png|.jpeg|.jpg')
uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    result = m.predict(opencv_image)

    if result['classes'] is not None:
        classes = [x for x in result['classes']] 
        st.text(result['classes'].values())
    else:
        None




# cp = CarsParser(m)
# path = './test_data'
# cp.parse(path)
# result = cp.get_report()
# result.to_csv('result.csv', index=False)
# print(result)