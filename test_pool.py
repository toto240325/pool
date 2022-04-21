"""
Unit tests module for this project
Just run like this :
1. to test with the local utils :
export PYTHONPATH=/home/toto/utils ; cd ~/pool ; venv ; python test_pool.py
2. to test wiht the production utils :
cd ~/pool ; venv ; python test_pool.py
"""

import unittest
import pool

class Testpool(unittest.TestCase):
    def test_values(self):
        #self.assertRaises(ValueError,friend, "wrong value for array ???"):
        pass
            
    def Test_assert_equals(self,p1,p2):
        self.assertEqual(p1,p2)
        
    # def test_pool_ps4_OK(self):        
    #     # self.Test_assert_equals(pool.main(),2)
    #     self.Test_assert_equals(pool.check_ps4("192.168.0.40"),True)

    # def test_pool_frigo_OK(self):        
    #     # self.Test_assert_equals(pool.main(),2)
    #     self.Test_assert_equals(pool.check_frigo(60*3),True)

    # def test_pool_frigo_NOK(self):        
    #     # self.Test_assert_equals(pool.main(),2)
    #     self.Test_assert_equals(pool.check_frigo(1),False)
        
    # def test_pool_full(self):        
    #     self.Test_assert_equals(pool.main(),None)
        

if __name__ == "__main__":
    unittest.main()
