#pyright: reportMissingImports=false

from enum import Enum

class Constants:

    class CDSConstants(Enum):

        TIME_FIELD = 'valid_time'
        LAT_FIELD = 'latitude'
        LON_FIELD = 'longitude'
        PRESSURE_FIELD = 'pressure_level'

    class Graphcast(Enum):

        TIME_FIELD = 'time'
        LAT_FIELD = 'latitude'
        LON_FIELD = 'longitude'
        PRESSURE_FIELD = 'level'
        BATCH_FIELD = 'batch'
