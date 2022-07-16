import configparser
import os

_CONFIG_FNAME = None
_config = configparser.ConfigParser()


def _save_changes():
    with open(_CONFIG_FNAME, 'w') as f:
        _config.write(f)


def load_config(fname: str):
    global _config
    global _CONFIG_FNAME
    _CONFIG_FNAME = fname
    _config.read(_CONFIG_FNAME)


class ConfigEntry:
    """Configuration entry class. Used to set and get configuration values.
    Attributes
    ----------
    key1: str
        Module or category name
    key2: str
        Entry name
    default: any
        Default value given by ctor
    type_ctor: any
        function called to get value from string repr
    Methods
    -------
    get:
        Get entry current value
    set:
        Set entry value (could be env or file)
    """

    def __init__(self, key1: str, key2: str, default: any = "", description: str = "", type_ctor=str):
        self.key1 = key1
        self.key2 = key2
        self.default = str(default)
        self.type_ctor = type_ctor
        self.description = description
        self.env_var_name = f"{self.key1}_{self.key2}".upper().replace('-', '_')

    def __repr__(self):
        return f"""ConfigEntry: {self.key1}/{self.key2}
    environment variable name: {self.env_var_name}
    value:                     {self.get()}
    description:               {self.description}"""

    def get(self):
        """Get configuration entry value. If a default is not provided then raise :class:`~speasy.config.exceptions.UndefinedConfigEntry`.
        Returns
        -------
        str:
            configuration value
        """
        if self.env_var_name in os.environ:
            return self.type_ctor(os.environ[self.env_var_name])
        if self.key1 in _config and self.key2 in _config[self.key1]:
            return self.type_ctor(_config[self.key1][self.key2])
        return self.type_ctor(self.default)

    def set(self, value: str):
        if self.env_var_name in os.environ:
            os.environ[self.env_var_name] = str(value)
        if self.key1 not in _config:
            _config.add_section(self.key1)
        _config[self.key1][self.key2] = str(value)
        _save_changes()


# ==========================================================================================
#                           ADD HERE CONFIG ENTRIES
# user can easily discover them with exact_laws.config.<completion>
# ==========================================================================================

preprocess_input_data_path = ConfigEntry('PREPROCESS_INPUT_DATA', 'path', '.')
preprocess_input_data_cycle = ConfigEntry('PREPROCESS_INPUT_DATA', 'cycle', 'cycle_0')
preprocess_input_data_sim_type = ConfigEntry('PREPROCESS_INPUT_DATA', 'sim_type', 'OCA_CGL2')

preprocess_output_data_path = ConfigEntry('PREPROCESS_OUTPUT_DATA', 'path', '.')
preprocess_output_data_name = ConfigEntry('PREPROCESS_OUTPUT_DATA', 'name', 'OCA_CGL2_cycle0_completeInc')
preprocess_output_data_reduction = ConfigEntry('PREPROCESS_OUTPUT_DATA', 'reduction', 4, type_ctor=int)

computation_output_path = ConfigEntry('COMPUTATION_OUTPUT', 'path', '.')
computation_output_name = ConfigEntry('COMPUTATION_OUTPUT', 'name', 'EL_logcyl40_cls100')

enabled_laws = ConfigEntry('COMPUTATION', 'laws', ['SS22IGyr'], type_ctor=eval)
enabled_terms = ConfigEntry('COMPUTATION', 'terms', ['flux_test'], type_ctor=eval)
enabled_quantities = ConfigEntry('COMPUTATION', 'quantities', [], type_ctor=eval)
use_reduced_datasets = ConfigEntry('COMPUTATION', 'use_reduced_datasets', True, type_ctor=eval)

with_mpi = ConfigEntry('COMPUTATION', 'with_mpi', False, type_ctor=eval)
numba_parallel = ConfigEntry('COMPUTATION', 'numba_parallel', False, type_ctor=eval)
compat_mode = ConfigEntry('COMPUTATION', 'compat_mode', False, type_ctor=eval)

nblayers = ConfigEntry('COMPUTATION', 'nblayers', 8, type_ctor=int)
nbbuff = ConfigEntry('COMPUTATION', 'nbbuff', 4, type_ctor=int)
save_frequency = ConfigEntry('COMPUTATION', 'save_frequency', -1, type_ctor=int)
restart_checkpoint = ConfigEntry('COMPUTATION', 'restart_checkpoint', "")

grid_n_max_scale = ConfigEntry('COMPUTATION_GRID', 'n_max_scale', 5, type_ctor=int)
grid_n_max_list = ConfigEntry('COMPUTATION_GRID', 'n_max_list', 3, type_ctor=int)
grid_kind = ConfigEntry('COMPUTATION_GRID', 'kind', 'cls')
grid_coords = ConfigEntry('COMPUTATION_GRID', 'coords', 'logcyl')

physical_params_di = ConfigEntry('PHYSICAL_PARAMS', 'di', 1, type_ctor=float)
