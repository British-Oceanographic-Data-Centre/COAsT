"""
Run this file from the main COAsT directory to generate a new unit_test_contents
file based on the imported modules in tests_to_do

If you have added new tests to the unit test, you should do this before pushing
to github.

The easiest way to do this is to copy and paste your new tests modules below
(or straight from unit_test.py).
"""

# Import modules, including unittest
import coast
import sys

sys.path.append("./unit_testing")
import logging

# Import tests to test. From a module (file) import a testCase class (See template)
from test_TEMPLATE import test_TEMPLATE
from test_gridded_initialisation import test_gridded_initialisation
from test_gridded_harmonics import test_gridded_harmonics
from test_general_utils import test_general_utils
from test_xesmf_convert import test_xesmf_convert
from test_diagnostic_methods import test_diagnostic_methods
from test_transect_methods import test_transect_methods
from test_object_manipulation import test_object_manipulation
from test_altimetry_methods import test_altimetry_methods
from test_tidegauge_methods import test_tidegauge_methods
from test_isobath_contour_methods import test_contour_f_methods
from test_isobath_contour_methods import test_contour_t_methods
from test_eof_methods import test_eof_methods
from test_profile_methods import test_profile_methods
from test_plot_utilities import test_plot_utilities
from test_stats_utilities import test_stats_utilities
from test_maskmaker_methods import test_maskmaker_methods
from test_climatology import test_climatology
from test_example_scripts import test_example_scripts
from test_WOD_read_data import test_WOD_read_data

# Open log file
log_file = open("unit_testing/unit_test.log", "w")  # Need log_file.close()
coast.logging_util.setup_logging(stream=log_file, level=logging.CRITICAL)

# Test list -- comment out ones you don't want maybe (or add your own)
tests_to_do = [
    # test_TEMPLATE,
    test_gridded_initialisation,
    test_general_utils,
    test_gridded_harmonics,
    test_xesmf_convert,
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
    test_WOD_read_data,
    test_example_scripts,
]

# Auto generate contents file. Define output file:
fn_contents = "./unit_testing/unit_test_contents.txt"

# Initialize counters
test_count = 1

# Open output file
with open(fn_contents, "w") as file:

    # Write title things
    file.write("     UNIT TEST CONTENTS FILE TEST    \n")
    file.write("\n")

    # Loop over tests in tests_to_do and get name of module as string
    for test in tests_to_do:
        test_name = test.__name__
        file.write("{0}. {1}\n".format(test_count, test_name))
        method_count = 97

        # Loop over methods in module. If begins with 'test_' then write to file
        for method in dir(test):
            if method[:5] != "test_":
                continue
            file.write("      {0}. {1}\n".format(chr(method_count), method[5:]))
            method_count = method_count + 1
        file.write("\n")
        test_count = test_count + 1
print("Written modules and methods to: \n \n")
print("{0}".format(fn_contents))
