# Author: ray
# Date: 4/16/24

import unittest

from util.model_until.model_loader import ModelLoader


class MyTestCaseModelLoader(unittest.TestCase):
    def test_read_json_and_create_namespace(self):
        file_path = "/Users/ray/rayfile/self-project/research_ml_visualization/hpc_sync_files/meta_info/model_meta_info/pure_sin_first_with_meta_script_20240330@03h10m09s_20240330@03h10m09s.json"
        nsp = ModelLoader.read_json_and_create_namespace(file_path)
        print(nsp)

    def test_model_load(self):
        model_id = "pure_sin_first_with_meta_script_20240330@03h09m59s_20240330@03h09m59s"
        ml = ModelLoader(model_id=model_id)
        ml.load_model()


if __name__ == '__main__':
    unittest.main()
