# Author: ray
# Date: 3/31/24

import unittest

from meta_info_process.meta_info_processor import MetaInfoProcessor


class MyTestCaseMetaInfoProcessor(unittest.TestCase):
    def test__load_all_model_info(self):
        mip = MetaInfoProcessor()
        mip._load_all_model_info()
        print(mip.model_mate_info_list)



if __name__ == '__main__':
    unittest.main()
