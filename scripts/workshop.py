import nvbs_models
import cv2
from cars_parser import CarsParser
import os
import streamlit as st
import numpy as np
import pandas as pd

m = nvbs_models.NvbsCarModel(
    classifier=nvbs_models.CarsClassifier('./model_weights/b4.pt'),
    detector=nvbs_models.CarsDetector('./model_weights/yolov5m.pt'),
    ocr=nvbs_models.PanelDigitsDetector('./model_weights/class_resnet18d.pt',
                                        './model_weights/bbox_model_resnet18d.pt')
    )

empty_messages = {
    'front': 'передней стороны',
    'f_right': 'правой передней стороны',
    'f_left': 'левой передней стороны',
    'left': 'левой стороны',
    'right': 'правой стороны',
    'back': 'задней стороны',
    'b_right': 'задней правой стороны',
    'b_left': 'задней левой стороны',
    'dirt': 'грязный',
    'dirty': 'грязный',
    'panel': 'панель',
    'interior': 'салон'}

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
        classes = pd.Series(list(result['classes'])).map(empty_messages)
        st.text('Атрибуты:')
        for x in classes.tolist(): 
            st.text(x)
        if result['mileage'] is not None:
            st.text('Пробег:')
            st.write(result['mileage'])
        else:
            None
    else:
        None

if st.button('Мне лень грузить картинку'):
    st.text(os.listdir(os.listdir(os.getcwd()+'/lazy_ass')))





# cp = CarsParser(m)
# path = './test_data'
# cp.parse(path)
# result = cp.get_report()
# result.to_csv('result.csv', index=False)
# print(result)