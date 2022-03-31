# SETTINGS

# Set the following to true to use the Coverage library to estimate unit_test
# coverage. You must have installed Coverage separately for this to work.
# You can do this using pip: pip install coverage.
# See docs website for more info. This will take longer to execute.
calculate_coverage = False

# Set the following to True to also auto-generate a new unit_test_contents file.
generate_unit_test_contents = False
fn_contents = "./unit_testing/unit_test_contents.txt"

# Set TEST imports and list of tests to do.
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
from test_tidegauge_methods import test_tidegauge_methods, test_tidegauge_analysis
from test_isobath_contour_methods import test_contour_t_methods, test_contour_f_methods
from test_eof_methods import test_eof_methods
from test_profile_methods import test_profile_methods
from test_plot_utilities import test_plot_utilities
from test_stats_utilities import test_stats_utilities
from test_maskmaker_methods import test_maskmaker_methods
from test_climatology import test_climatology
from test_example_scripts import test_example_scripts


# Test list -- comment out ones you don't want maybe (or add your own)
tests_to_do = [
    # test_TEMPLATE,
    test_xesmf_convert,
    test_gridded_initialisation,
    test_general_utils,
    test_gridded_harmonics,
    test_diagnostic_methods,
    test_transect_methods,
    test_object_manipulation,
    test_altimetry_methods,
    test_tidegauge_methods,
    test_tidegauge_analysis,
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




# UNIT TESTING CONTROL SCRIPT

# Import modules, including unittest
import unittest
import sys
import coast
sys.path.append("./unit_testing")
import logging

# Open log file
log_file = open("unit_testing/unit_test.log", "w")  # Need log_file.close()
coast.logging_util.setup_logging(stream=log_file, level=logging.CRITICAL)

# If this is enabled, import the library and start the coverage calculation
if calculate_coverage:
    import coverage
    cov = coverage.Coverage()
    cov.start()

# Create suite - this is a collection of tests, defined by classes
# Add tests to test to the suite -- Add in a line for each suite
# suite.addTest(unittest.makeSuite(test_TEMPLATE))
suite = unittest.TestSuite()
for test in tests_to_do:
    suite.addTest(unittest.makeSuite(test))

# Run test suite. Some different verbosity options available here.
unittest.TextTestRunner(verbosity=2).run(suite)

if calculate_coverage:
    print(' ')
    print('COVERAGE REPORT: ')
    cov.stop()
    cov.save()
    cov.report(omit="unit_testing/test_*")

# Generate unit_test_contents
if generate_unit_test_contents:
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
    print(" ")
    print("Written modules and methods to:")
    print("          {0}".format(fn_contents))
