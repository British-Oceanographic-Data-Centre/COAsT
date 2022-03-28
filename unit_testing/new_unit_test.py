# Import modules, including unittest
import unittest
import coast
import sys
import os.path as path

# Import tests to test (I have only written one for now. This is a module)
import test_xesmf_convert

# Create suite - this is a collection of tests, defined by classes
suite = unittest.TestSuite()

# Add tests to test to the suite -- Add in a line for each suite
suite.addTest(unittest.makeSuite(test_xesmf_convert.test_xesmf_convert))
# suite.addTest(...etc)

# Run test suite. Some different verbosity options available here.
unittest.TextTestRunner(verbosity=3).run(suite)
