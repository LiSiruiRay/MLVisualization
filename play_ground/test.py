# Author: ray
# Date: 4/16/24

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # self.assertEqual(True, False)  # add assertion here
        # import pkg_resources
        # pkg_resources.require("FEDformer-installable")  # Should not throw an error if installed
        #
        # # Test importing
        # from FEDformer-installable.models.autoformer import Autoformer
        import sys
        sys.path.append('/path/to/your/virtualenv/lib/python3.10/site-packages')

        from FEDformerInstallable.models.autoformer import Autoformer
        # from FEDformerInstallable.models import Autoformer
        # print(Autoformer)
        # import FEDformer_installable

if __name__ == '__main__':
    unittest.main()
