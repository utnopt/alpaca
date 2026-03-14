# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from alpaca.utils.logger import logger
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf


def check_pwl_method_for_mpip_feature(func):
    """Decorator to ensure the correct PWL method before executing the function."""

    def wrapper(self, *args, **kwargs):
        mpip = next(iter(self.mpip_handler.mpip_dict.values()))
        if mpip.pwl_method != lsf.pwl_method_multiple_choice():
            logger.warning(
                lsf.warning_not_implemented_mpip_feature_for_pwl_method(mpip.pwl_method)
            )
            return
        func(self, *args, **kwargs)

    return wrapper
