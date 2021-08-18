import nvbs_models
import cv2
from cars_parser import CarsParser
import os
import streamlit as st
import numpy as np
import pandas as pd
from streamlit import caching
import random

m = nvbs_models.NvbsCarModel(
    classifier=nvbs_models.CarsClassifier('./model_weights/b4.pt'),
    detector=nvbs_models.CarsDetector('./model_weights/yolov5m.pt'),
    ocr=nvbs_models.PanelDigitsDetector('./model_weights/class_resnet18d.pt',
                                        './model_weights/bbox_model_resnet18d.pt')
    )

empty_messages = {
    'front': 'передняя сторона',
    'f_right': 'правая передняя сторона',
    'f_left': 'левая передняя сторона',
    'left': 'левая сторона',
    'right': 'правая сторона',
    'back': 'задняя сторона',
    'b_right': 'задняя правая сторона',
    'b_left': 'задняя левая сторона',
    'dirt': 'грязный',
    'dirty': 'грязный',
    'panel': 'панель',
    'interior': 'салон',
    'trunk': 'багажник',
    'labeled': 'наклейка присутствует',
    'damaged': 'повреждена, нарушена целостность наклейки/отсутствует наклейка',
    'rubbish': 'внутри находится мусор'}

st.title('НВБС')
st.subheader('Инструмент для определения состояний автомобилей и показаний одометра')


st.text('Загрузите картинку с автомобилем сюда в формате .png|.jpeg|.jpg')
uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])

def on_download():
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    result = m.predict(opencv_image)

    st.text(result)

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

if uploaded_file is not None:
    # Convert the file to an opencv image.
    on_download()


def random_show():
    random_num = random.choice(os.listdir(os.getcwd()+'/lazy_ass'))
    opencv_image = cv2.imread(os.getcwd()+'/lazy_ass/' + random_num)
    w, h = round(opencv_image.shape[0]*0.4), round(opencv_image.shape[1]*0.4)

    to_show = cv2.resize(opencv_image, (w, h))

    st.image(opencv_image, channels="BGR")

    result = m.predict(opencv_image)
    
    if result['classes'] is not None:
        classes = pd.Series(list(result['classes'])).map(empty_messages)
        st.text('Атрибуты:')
        for x in classes.tolist(): 
            st.text(x)
        
        try:
            if result['mileage'] is not None:
                st.text('Пробег:')
                st.write(result['mileage'])
            else:
                None
        except:
            None
    else:
        None


if st.button('Мне повезёт!'):
    random_show()



caching.clear_cache()



# cp = CarsParser(m)
# path = './test_data'
# cp.parse(path)
# result = cp.get_report()
# result.to_csv('result.csv', index=False)
# print(result)