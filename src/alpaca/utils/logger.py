"""
always use this logger so it's package-wise configurable
(it's a separate module to avoid circular imports)

@authors: hamm,
"""

import logging

logger = logging.getLogger("my_project")
