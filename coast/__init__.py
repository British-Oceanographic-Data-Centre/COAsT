from .data.coast import Coast, setup_dask_client
from ._utils.mask_maker import MaskMaker
from .data.gridded import Gridded
from .diagnostics.transect import Transect, TransectF, TransectT
from .diagnostics.contour import Contour, ContourF, ContourT
from .diagnostics.eof import compute_eofs, compute_hilbert_eofs
from .diagnostics.internal_tide import InternalTide
from .diagnostics.climatology import Climatology
from ._utils import logging_util, general_utils, plot_util, crps_util
from .data.index import Indexed
from .data.profile import Profile
from .diagnostics.profile_analysis import ProfileAnalysis
from .data.track import Track
from .data.lagrangian import Lagrangian
from .data.oceanparcels import Oceanparcels
from .data.glider import Glider
from .data.argos import Argos
from .data.altimetry import Altimetry
from .data.timeseries import Timeseries
from .data.tidegauge import Tidegauge
from .diagnostics.tidegauge_analysis import TidegaugeAnalysis
from .data.config_parser import ConfigParser
from ._utils.xesmf_convert import xesmf_convert
from ._utils.process_data import Process_data
from .data.opendap import OpendapInfo
from .data.copernicus import Copernicus, Product

# Set default for logging level when coast is imported
import logging

logging_util.setup_logging(level=logging.CRITICAL)
