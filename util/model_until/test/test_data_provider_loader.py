# Author: ray
# Date: 4/16/24

import unittest

from util.model_until.data_provider_loader import DataProviderLoader


class MyTestCaseDataProviderLoader(unittest.TestCase):
    def test_load_load_data_provider(self):
        model_id = "pure_sin_first_with_meta_script_20240330@03h09m59s_20240330@03h09m59s"
        dpl = DataProviderLoader(model_id=model_id)
        dpl.load_load_data_provider()
        print(f"result: {dpl.data_set}")


if __name__ == '__main__':
    unittest.main()
