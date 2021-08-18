import pandas as pd
import os
import cv2
import hashlib
from pathlib import Path
from tqdm import tqdm

class CarsParser():
    """Обертка для моделей, читает отдельные файлы или директории и пишет
    результат в хранилище. Проверяет, был ли данный файл считан ранее."""
    empty_messages = {
        'front': 'передней стороны',
        'f_right': 'правой передней стороны',
        'f_left': 'левой передней стороны',
        'left': 'левой стороны',
        'right': 'правой стороны',
        'back': 'задней стороны',
        'b_right': 'задней правой стороны',
        'b_left': 'задней левой стороны'}

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.storage_path = './storage.csv'
        self._read_storage()

    def _read_storage(self):
        try:
            self.storage = pd.read_csv(self.storage_path)
        except FileNotFoundError:
            print('Storage file not found, create a new one.')
            columns = [
                'city',
                'number',
                'date',
                'tags',
                'mileage',
                'md5',
                'path'
                ]
            self.storage = pd.DataFrame(columns=columns)

    def classificate(self, path, params):
        hash_md5 = self.md5(path)
        if self._is_new_image(hash_md5):
            img = cv2.imread(path)
            try:
                labels = self.ensemble.predict(img)
                self._insert_in_storage(labels, params, hash_md5, path)
            except Exception as err:
                print('Cant classificate image:', err)

    def md5(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_new_image(self, hash_md5):
        """Классифицировался ли данный файл раньше."""
        return hash_md5 not in self.storage['md5'].values

    def _insert_in_storage(self, labels, params, hash_md5, path):
        row = [
            params['city'],
            params['number'],
            params['date'],
            ','.join(labels['classes']),
            labels.get('mileage'),
            hash_md5,
            path
            ]
        self.storage.loc[len(self.storage), :] = row


    def save_storage(self):
        self.storage.to_csv(self.storage_path, index=False)

    def _scan_dir(self, path):
        folder = []
        for i in os.walk(str(path)):
            folder.append(i)

        result = 0
        for address, dirs, files in folder:
            for file in files:
                result += 1
        return result

    def parse(self, folder):
        folder = Path(folder).absolute()
        total_files = self._scan_dir(folder)

        with tqdm(total=total_files) as pbar:
            cities = os.listdir(str(folder))
            for city in cities:
                path = folder.joinpath(city)
                car_numbers = os.listdir(str(path))
                for c_num in car_numbers:
                    path = folder.joinpath(city, c_num)
                    dates = os.listdir(str(path))
                    for d in dates:
                        path = folder.joinpath(city, c_num, d)
                        params = {
                            'city': city,
                            'number': c_num,
                            'date': d
                            }
                        files = os.listdir(str(path))
                        for file in files:
                            path = folder.joinpath(city, c_num, d, file)
                            if path.is_file():
                                self.classificate(str(path.absolute()), params)
                                pbar.update(1)
        self.save_storage()

    def _get_mileage_msg(self, subset):
        mileage = subset.loc[subset['mileage'].notna(), 'mileage']
        if len(mileage) == 0:
            mileage_msg = 'Нет показаний одометра'
        elif len(mileage) > 1:
            mileage_msg = 'Неоднозначные показания одометра'
        else:
            mileage_msg = f'{int(mileage.values[0])} км.'
        return mileage_msg

    def _get_body_msg(self, subset):
        body_full = ''
        for ix in subset.index:
            tags = subset.loc[ix, 'tags'].split(',')
            if 'damaged' in tags:
                for tag in self.empty_messages:
                    if tag in tags:
                        body_full += f'Есть повреждения {self.empty_messages[tag]}. '
        body_full = body_full.strip() if body_full != '' else 'Повреждений нет.'
        return body_full

    def _get_position_msg(self, subset, tags):
        position = ''
        for tag in self.empty_messages:
            if tag not in tags:
                position += f'Нет фото {self.empty_messages[tag]}. '
        position = position.strip() if position != '' else 'Все фото есть.'
        return position

    def _get_body_dirt_msg(self, subset, tags):
        msg = 'Чистая'
        for tag in subset['tags'].values:
            for t in ['front', 'back', 'left', 'right']:
                if t in tag and 'dirty' in tag:
                    msg = 'Грязная'
        return msg

    def _get_interior_dirt_msg(self, tags):
        return 'Грязный' if 'dirt' in tags else 'Чистый'

    def _get_interior_rubbish_msg(self, tags):
        return 'С вещами' if 'rubbish' in tags else 'Без вещей'

    def _get_trunk_msg(self, subset):
        for tag in subset['tags'].values:
            if 'trunk' in tag:
                if 'dirty' in tag:
                    msg = 'В багажнике посторонние предметы/грязь.'
                else:
                    msg = 'Багажник чист.'
                return msg

    def get_report(self):
        """Собрать человекочитаемую таблицу-отчет."""
        cols = ['city', 'number', 'date']
        result = []
        for ix, subset in self.storage.groupby(by=cols):
            number = ix[1]
            city = ix[0]
            date = ix[2]
            tags = set(','.join(subset['tags'].values).split(','))

            body_dirt = self._get_body_dirt_msg(subset, tags)
            interior_dirt = self._get_interior_dirt_msg(tags)
            interior_rubbish = self._get_interior_rubbish_msg(tags)
            body_full = self._get_body_msg(subset)
            mileage_msg = self._get_mileage_msg(subset)
            position = self._get_position_msg(subset, tags)
            trunk_msg = self._get_trunk_msg(subset)

            result_row = [number, city, date, position, body_dirt, body_full,
                          interior_dirt, interior_rubbish, trunk_msg,
                          mileage_msg]
            result.append(result_row)
        columns = [
            'Номер машины',
            'Город',
            'Дата',
            'Положение',
            'Состояние (чистая)',
            'Состояние (битая)',
            'Салон (чистый)',
            'Салон (предметы)',
            'Состояние багажника',
            'Пробег']
        result = pd.DataFrame(result, columns=columns)
        return result
