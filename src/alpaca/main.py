# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.locatelli.stair_locatelli as slo
import alpaca.settings as s
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class Alpaca:
    """Main Alpaca class."""

    def __init__(self, settings_path: str | None = None) -> None:
        """Initializes the Alpaca class."""
        ut_io.config_console_logger("INFO")
        config_dict = (
            ut_io.read_config_file(settings_path) if settings_path is not None else {}
        )
        self.user_settings = s.UserSettings(config_dict)
        self.model_data = mda.ModelData(self.user_settings)
        self.stair_locatelli: slo.StairLocatelli | None = None
        self.solver: slv.Solver | None = None
        self.runtime: int | None = None

    @staticmethod
    def configure_logging(path: str, level: str = "INFO") -> None:
        """Configures logging settings."""
        ut_io.config_console_logger(level)
        ut_io.config_file_logger(path, level)

    def customize_settings(self, settings_import: str | dict) -> None:
        """Customizes settings from a given path."""
        if isinstance(settings_import, dict):
            new_settings = s.UserSettings(settings_import)
        else:
            new_settings = s.UserSettings(ut_io.read_config_file(settings_import))
        self.user_settings.update_from_other(new_settings)

    def build_pwl_relaxation_solver(self) -> None:
        """Builds the PWL relaxation solver."""
        self.model_data.build_pwl_relaxation_model()

        if self.user_settings.feature_stair_locatelli:
            self.stair_locatelli = slo.StairLocatelli(self.model_data)

        external_solver = mm.MIPModel(
            self.model_data,
            nonlinear=self.user_settings.pwl_method == lsf.pwl_method_none(),
            bilinear=self.user_settings.bilinear_handling == 3,
        )

        self.solver = slv.Solver(external_solver)

        if self.user_settings.feature_mpip:
            if self.user_settings.pwl_method == lsf.pwl_method_none():
                logger.warning(lsf.warning_mpip_features_disabled_for_pwl_method_none())
            else:
                mpip_handler = mph.MPIPHandler(self.model_data)
                mpip_separation_handler = msh.MPIPSeparationHandler(
                    mpip_handler, external_solver.opt_model
                )
                self.solver.mpip_separation_handler = mpip_separation_handler

    def solve(self):
        """Solves the optimization problem."""
        self.runtime = self.solver.solve_instance()
        logger.info(lsf.info_optimization_finished(self.runtime))


def read_model_from_osil(path: str) -> Alpaca:
    """Reads and parses a model from an OSIL file."""
    alp = Alpaca()
    alp.model_data.read_model_from_osil_data(path)
    return alp
