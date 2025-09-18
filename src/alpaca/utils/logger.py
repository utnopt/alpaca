"""
always use this logger so it's package-wise configurable
(it's a separate module to avoid circular imports)

@authors: hamm, kuen
"""

import logging

from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf

logger = logging.getLogger(lsf.project_name())
