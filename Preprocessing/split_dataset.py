 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Preprocessing.preprocessing import *
from configuration import meta
from pprint import pprint

import json


#########################################################################################################################
# Create subject list and after shuffling it, split to train, valid and test sets
#########################################################################################################################
class JsonFoldList(MetaParameters):
    def __init__(self):
        super(MetaParameters, self).__init__()
        self.json_file_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_folds_list.json'
        self.folds_dict = self.choose_folds_list

    @property
    def dataset_list(self):
        dataset_list = ReadImages(f'{self.ORIGS_DIR}').get_dataset_list()

        try:
            dataset_list.remove('.DS_Store')
        except ValueError:
            pass

        random.shuffle(dataset_list)

        return dataset_list

    @property
    def json_dump(self):
        with open(self.json_file_path, "w") as fdct:
            json.dump(self.folds_dict, fdct) # записываем структуру в файл

    @property
    def choose_folds_list(self):
        dataset_size = len(self.dataset_list)
        test_list = self.dataset_list[round(0.8 * dataset_size):]
        train_list  = list(set(self.dataset_list) - set(test_list))

        train_dataset_size = len(train_list)

        valid_list_01 = train_list[round(0.8 * train_dataset_size):]
        train_list_01 = list(set(train_list) - set(valid_list_01))

        valid_list_02 = train_list[round(0.6 * train_dataset_size):round(0.8 * train_dataset_size)]
        train_list_02 = list(set(train_list) - set(valid_list_02))

        valid_list_03 = train_list[round(0.4 * train_dataset_size):round(0.6 * train_dataset_size)]
        train_list_03 = list(set(train_list) - set(valid_list_03))

        valid_list_04 = train_list[round(0.2 * train_dataset_size):round(0.4 * train_dataset_size)]
        train_list_04 = list(set(train_list) - set(valid_list_04))

        valid_list_05 = train_list[:round(0.2 * train_dataset_size)]
        train_list_05 = list(set(train_list) - set(valid_list_05))

        folds_dict = {
                    'train_list_01': train_list_01, 'valid_list_01': valid_list_01,
                    'train_list_02': train_list_02, 'valid_list_02': valid_list_02,
                    'train_list_03': train_list_03, 'valid_list_03': valid_list_03,
                    'train_list_04': train_list_04, 'valid_list_04': valid_list_04,
                    'train_list_05': train_list_05, 'valid_list_05': valid_list_05,
                    'train_list_full': train_list, 'valid_list_full': test_list,
                    'test_list': test_list,
                    }

        return folds_dict

    @property
    def create_folds_list(self):
        if not os.path.exists(f'{self.DATASET_DIR}{self.DATASET_NAME}_folds_list.json'):
            self.json_dump

    def json_load(self, list_name: str):
        try:
            with open(self.json_file_path, "r") as fdct:
                folds_dict = json.load(fdct)

                if list_name == 'test_list':
                    dataset_list = folds_dict[list_name]
                else:
                    dataset_list = folds_dict[f'{list_name}_{self.FOLD_NAME}']

                return dataset_list
        except:
            print('ERROR!!! LOOK AT JSON FILE IN DATASET DIRECTORY')

    def load_dataset_list(self, list_name: str):
        return self.json_load(list_name)

    def pprint(self, list_name: str):
        try:
            dataset_list = self.load_dataset_list(list_name)

            pprint(f'{list_name} = {dataset_list}')

        except:
            pprint('ERROR!!! LOOK AT JSON FILE IN DATASET DIRECTORY')



if __name__ == '__main__':
    jsnlst = JsonFoldList()
    jsnlst.create_folds_list
    test_list = jsnlst.load_dataset_list('test_list')


