 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os



class FileDirectoryWorker():
    # def __init__(self):
    #     self.project_name = project_name
    #     self.results = results

    def log_stats(self, project_name: str, results = None):
        file = open(f'{project_name}_log.txt', 'a')
        file.write(results + "\n")
        file.close()

    def create_dir(self, project_name: str):
        os.makedirs(project_name, exist_ok = True)

    def create_dir_log(self, project_name: str):
        os.makedirs(project_name, exist_ok = True)

        if not os.path.exists(f'{project_name}_log.txt'):
            my_log = open(f'{project_name}_log.txt', 'a')
            my_log.close()
        

