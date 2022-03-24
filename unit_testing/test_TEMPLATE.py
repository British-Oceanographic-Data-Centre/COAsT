'''
TEMPLATE testing file, usin unittest package.
Please save this file with a name starting with "test_".
TestCase classes have a whole bunch of methods available to them. Some of them
are showcased below. You can also add your own methods to them. Anything you
want tested by the unit testing system should start with "test_".

For more info on assert test cases, see:
    https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertTrue
'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import os.path as path

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files

# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_TEMPLATE(unittest.TestCase):
    
    def setUp(self):
        # This is called at the beginning of every test_ method. Optional.
        return
    
    def tearDown(self):
        # This is called at the end of every test_ method. Optional.
        return
    
    @classmethod
    def setUpClass(cls):
        # This is called at the beginning of every test CLASS. Optional
        return
    
    @classmethod
    def tearDownClass(cls):
        # This is called at the end of every test CLASS. Optional.
        return
    
    # TEST METHODS
    def test_successful_method(self):
        # Test things. Use assert methods from unittest. Some true tests:
        self.assertTrue(1==1, "Will not see this.")
        self.assertEqual(1,1, "Will not see this.")
        self.assertIsNone(None, "Will not see this.")
        
        
    def test_bad_method(self):
       # Test things. Use assert methods from unittest. This will fail
        self.assertTrue(1==2, "1 does not equal 2")
        self.assertEqual(1,2, "1 does not equal 2.")
        self.assertIsNone(1, "1 does not none.")
        
    def test_with_subtests(self):
        # Sometimes you might want to do a test within a parameter loop or
        # a bunch of sequential tests (so you don't have to read data 
        # repeatedly). Subtest can be used for this as follows:
            
        with self.subTest("First subtest."):
            a = 1
            self.assertEqual(a,1,"Will not see this.")
            
        with self.subTest("Second subtest"):
            b = 1
            self.assertEqual(a,b, "will not see this")
            
        with self.subTest("Third subtest"):
            c =  50000000
            self.assertAlmostEqual(a, c, msg="0 is not almost equal to 50000000")

