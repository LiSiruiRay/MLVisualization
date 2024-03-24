# Author: ray
# Date: 3/24/24

import unittest

from util.model_meta_info_reading import get_model_list_from_folder_reading


class MyTestCaseModelMetaInfo(unittest.TestCase):
    def test_getting_list_models(self):
        model_list = get_model_list_from_folder_reading()
        print(model_list)


if __name__ == '__main__':
    unittest.main()
