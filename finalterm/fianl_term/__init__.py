__version__ = '0.0.1' #241211

import warnings
warnings.filterwarnings('ignore')

from .output_histogram import * #noqa
from .small_dist_reduction import * #noqa
from .PLS import * #noqa
from .kernelRidge import * #noqa
from .random_forest import * #noqa
from .gaussian_process_regression import * #noqa
from .gaussian_random_projection import * #noqa
from .interaction_regression import * #noqa
from .kernelPCA import * #noqa
from .linear_PCA import * #noqa
from .label_encoding import * #noqa