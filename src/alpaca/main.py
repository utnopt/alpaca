# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.locatelli.ortho_locatelli as slo
import alpaca.settings as s
import alpaca.stats.statistics as ass
from alpaca.utils import inout as ut_io
from alpaca.utils.logger import logger
from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf


class Alpaca:
    """The main entry point for the Alpaca optimization framework.

    This class orchestrates the loading of model data, configuration of settings,
    construction of the piecewise linear (PWL) relaxation solver, and the
    execution of the optimization process.

    Attributes:
        user_settings (s.UserSettings): Configuration object for the solver behavior.
        model_data (mda.ModelData): Data structure holding the optimization model.
        ortho_locatelli (slo.OrthoLocatelli | None): Handler for Ortho-Locatelli features.
        solver (slv.Solver | None): The initialized solver instance.
    """

    def __init__(self, settings_path: str | None = None) -> None:
        """Initializes the Alpaca instance with optional external settings.

        Args:
            settings_path: Filesystem path to a configuration file (e.g., JSON/YAML).
                If None, default settings are used.
        """
        ut_io.config_console_logger("INFO")
        config_dict = (
            ut_io.read_config_file(settings_path) if settings_path is not None else {}
        )
        self.statistics = ass.Statistics(self)
        self.user_settings = s.UserSettings(config_dict)
        self.model_data = mda.ModelData(self.user_settings)
        self.ortho_locatelli: slo.OrthoLocatelli | None = None
        self.solver: slv.Solver | None = None
        self.obbt_variable_bounds: dict[str, tuple[float, float]] = {}
        self.locatelli_vertices: dict[str, list[tuple[float, float]]] = {}

    def configure_logging(self, path: str, level: str = "INFO") -> None:
        """Configures global logging for both console and file output.

        Args:
            path: Path where the log file should be saved.
            level: Logging threshold level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.statistics.log_file_path = path
        ut_io.config_file_logger(path, level)

    def customize_settings(self, settings_import: str | dict) -> None:
        """Merges new settings into the current user configuration.

        Args:
            settings_import: Either a dictionary of settings or a path to a
                configuration file to be parsed.
        """
        if isinstance(settings_import, dict):
            new_settings = s.UserSettings(settings_import)
        else:
            self.statistics.config_name = settings_import.split(lsf.path_separator())[
                -1
            ].replace(lsf.json_file_suffix(), lsf.empty_string())
            new_settings = s.UserSettings(ut_io.read_config_file(settings_import))
        self.user_settings.update_from_other(new_settings)

    def build_pwl_relaxation_solver(self) -> None:
        """Constructs the solver using a Piecewise Linear (PWL) relaxation model.

        This method initializes the internal model data, configures Ortho-Locatelli
        heuristics if enabled, sets up the external MIP solver, and handles
        MPIP (Mixed-Integer Programming Partitioning) separation logic based
        on the user settings.
        """
        self.statistics.build_started()
        self.statistics.track_statistics_original_model()
        obbt_variable_bounds = (
            self.obbt_variable_bounds if self.obbt_variable_bounds else None
        )
        self.obbt_variable_bounds = self.model_data.build_pwl_relaxation_model(
            obbt_variable_bounds=obbt_variable_bounds
        )
        self.statistics.track_statistics_pwl_model()

        if self.user_settings.feature_ortho_locatelli:
            locatelli_vertices = (
                self.locatelli_vertices if self.locatelli_vertices else None
            )
            self.ortho_locatelli = slo.OrthoLocatelli(
                self.model_data, locatelli_vertices=locatelli_vertices
            )
            self.locatelli_vertices = self.ortho_locatelli.locatelli_vertices

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
                self.statistics.track_statistics_mpip()
        self.statistics.build_finished()

    def solve(self):
        """Executes the optimization process for the built model."""
        if self.solver is None:
            raise RuntimeError(
                "Solver is not initialized. "
                "Call 'build_pwl_relaxation_solver' before calling 'solve'."
            )
        runtime = self.solver.solve_instance()
        logger.info(lsf.info_optimization_finished(runtime))
        self.statistics.get_solver_information_from_external_solver_log()
        self.statistics.track_statistics_ortho_locatelli()
        if (
            self.user_settings.feature_mpip
            and self.user_settings.pwl_method != lsf.pwl_method_none()
        ):
            self.statistics.track_statistics_mpip_separation()


def read_model_from_osil(path: str) -> Alpaca:
    """Creates an Alpaca instance and populates it with data from an OSIL file.

    Args:
        path: Filesystem path to the .osil file.

    Returns:
        Alpaca: An initialized instance with loaded model data.
    """
    alp = Alpaca()
    alp.model_data.read_model_from_osil_data(path)
    alp.statistics.instance_name = path.split(lsf.path_separator())[-1].replace(
        lsf.osil_file_suffix(), lsf.empty_string()
    )
    return alp
