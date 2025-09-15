# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""

import alpaca.pwl.pwl_method as pwm


class DeltaMethod(pwm.PWLMethod):
    """Represents the delta method for pwl."""

    def _represent_domain(self):
        """Represent the domain of a variable using the delta method."""
        raise NotImplementedError("Delta method not implemented yet.")
