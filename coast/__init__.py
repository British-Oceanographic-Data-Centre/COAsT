from .coast import Coast
from .coast import setup_dask_client
from .mask_maker import MaskMaker
from .gridded import Gridded
from .transect import Transect, TransectF, TransectT
from .contour import Contour, ContourF, ContourT
from .eof import compute_eofs, compute_hilbert_eofs
from .internal_tide import InternalTide
from .climatology import Climatology
from . import logging_util
from . import general_utils
from . import plot_util
from . import crps_util
from .index import Indexed
from .profile import Profile
from .track import Track
from .lagrangian import Lagrangian
from .oceanparcels import Oceanparcels
from .glider import Glider
from .argos import Argos
from .altimetry import Altimetry
from .timeseries import Timeseries
from .tidegauge import Tidegauge
from .tidegauge_multiple import TidegaugeMultiple
from .config_parser import ConfigParser

# Set default for logging level when coast is imported
import logging

logging_util.setup_logging(level=logging.CRITICAL)
