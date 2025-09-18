# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""

import alpaca.pwl.pwl_method as pwm
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class DeltaMethod(pwm.PWLMethod):
    """Represents the delta method for pwl."""

    def _represent_domain(self):
        """Represent the domain of a variable using the delta method."""
        raise NotImplementedError(lsf.error_not_implemented_delta_method())
