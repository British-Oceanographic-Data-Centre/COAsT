# Import modules, including unittest
import unittest
import coast
import sys

sys.path.append("./unit_testing")
import logging

# Import tests to test. From a module (file) import a testCase class (See template)
from test_TEMPLATE import test_TEMPLATE
from test_xesmf_convert import test_xesmf_convert
from test_gridded_initialisation import test_gridded_initialisation
from test_gridded_harmonics import test_gridded_harmonics
from test_general_utils import test_general_utils
from test_diagnostic_methods import test_diagnostic_methods
from test_transect_methods import test_transect_methods
from test_object_manipulation import test_object_manipulation
from test_altimetry_methods import test_altimetry_methods
from test_tidegauge_methods import test_tidegauge_methods
from test_isobath_contour_methods import test_contour_t_methods, test_contour_f_methods
from test_eof_methods import test_eof_methods
from test_profile_methods import test_profile_methods
from test_plot_utilities import test_plot_utilities
from test_stats_utilities import test_stats_utilities
from test_maskmaker_methods import test_maskmaker_methods
from test_climatology import test_climatology
from test_example_scripts import test_example_scripts
from test_process_data import test_process_data_methods

# Open log file
log_file = open("unit_testing/unit_test.log", "w")  # Need log_file.close()
coast.logging_util.setup_logging(stream=log_file, level=logging.CRITICAL)

# Test list -- comment out ones you don't want maybe (or add your own)
tests_to_do = [
    # test_TEMPLATE,
    test_process_data_methods,
    test_xesmf_convert,
    test_gridded_initialisation,
    test_general_utils,
    test_gridded_harmonics,
    test_diagnostic_methods,
    test_transect_methods,
    test_object_manipulation,
    test_altimetry_methods,
    test_tidegauge_methods,
    test_eof_methods,
    test_contour_f_methods,
    test_contour_t_methods,
    test_profile_methods,
    test_plot_utilities,
    test_stats_utilities,
    test_maskmaker_methods,
    test_climatology,
    test_example_scripts,
]

# Create suite - this is a collection of tests, defined by classes
# Add tests to test to the suite -- Add in a line for each suite
# suite.addTest(unittest.makeSuite(test_TEMPLATE))
suite = unittest.TestSuite()
for test in tests_to_do:
    suite.addTest(unittest.makeSuite(test))

# Run test suite. Some different verbosity options available here.
unittest.TextTestRunner(verbosity=2).run(suite)
