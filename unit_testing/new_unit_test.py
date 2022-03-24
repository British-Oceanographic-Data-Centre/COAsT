# Import modules, including unittest
import unittest
import coast
import sys
sys.path.append("./unit_testing")
import os.path as path

# Import tests to test (I have only written one for now. This is a module)
from test_TEMPLATE                import test_TEMPLATE
from test_xesmf_convert           import test_xesmf_convert
from test_gridded_initialisation  import test_gridded_initialisation
from test_gridded_harmonics       import test_gridded_harmonics
from test_general_utils           import test_general_utils
from test_diagnostic_methods      import test_diagnostic_methods
from test_transect_methods        import test_transect_methods
from test_object_manipulation     import test_object_manipulation
from test_altimetry_methods       import test_altimetry_methods
from test_tidegauge_methods       import test_tidegauge_methods
from test_isobath_contour_methods import test_isobath_contour_methods
from test_eof_methods             import test_eof_methods
from test_profile_methods         import test_profile_methods
from test_plot_utilities          import test_plot_utilities
from test_stats_utilities         import test_stats_utilities
from test_maskmaker_methods       import test_maskmaker_methods
from test_climatology             import test_climatology
from test_example_scripts         import test_example_scripts

# Create suite - this is a collection of tests, defined by classes
suite = unittest.TestSuite()

# Add tests to test to the suite -- Add in a line for each suite
#suite.addTest(unittest.makeSuite(test_TEMPLATE))
#suite.addTest(unittest.makeSuite( test_xesmf_convert ))
#suite.addTest(unittest.makeSuite( test_gridded_initialisation ))
#suite.addTest(unittest.makeSuite( test_general_utils ))
#suite.addTest(unittest.makeSuite( test_gridded_harmonics ))
#suite.addTest(unittest.makeSuite( test_diagnostic_methods ))
#suite.addTest(unittest.makeSuite( test_transect_methods ))
#suite.addTest(unittest.makeSuite( test_object_manipulation ))
#suite.addTest(unittest.makeSuite( test_altimetry_methods ))
#suite.addTest(unittest.makeSuite( test_tidegauge_methods ))
#suite.addTest(unittest.makeSuite( test_eof_methods ))
#suite.addTest(unittest.makeSuite( test_isobath_contour_methods ))
#suite.addTest(unittest.makeSuite( test_profile_methods ))
#suite.addTest(unittest.makeSuite( test_plot_utilities ))
#suite.addTest(unittest.makeSuite( test_stats_utilities ))
#suite.addTest(unittest.makeSuite( test_maskmaker_methods ))
#suite.addTest(unittest.makeSuite( test_climatology ))
suite.addTest(unittest.makeSuite( test_example_scripts ))

# Run test suite. Some different verbosity options available here.
unittest.TextTestRunner(verbosity=2).run(suite)
