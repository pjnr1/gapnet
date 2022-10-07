from enum import Enum
import os.path
import glob


class LocationNames(Enum):
    DTU_HPC_JL = 1
    LOCAL_JL_1 = 2


home_to_location = {
    '/zhome/e8/0/78560': LocationNames.DTU_HPC_JL,
    '/Users/jenslindahl': LocationNames.LOCAL_JL_1,
}

thesis_folder_locations = {
    LocationNames.DTU_HPC_JL: '/zhome/e8/0/78560/msc_thesis',
    LocationNames.LOCAL_JL_1: '/Users/jenslindahl/repos/msc_thesis',
}

location = home_to_location[os.getenv('HOME')]


def get_thesis_folder():
    return thesis_folder_locations[location]


def get_plot_output_folder():
    return os.path.join(get_thesis_folder(), 'plots')


def get_datafolder():
    return os.path.join(thesis_folder_locations[location], 'data')


def get_model_output_folder():
    return os.path.join(get_datafolder(), 'model_output')


def get_external_results_folder(experiment: str):
    return os.path.join(get_datafolder(), 'external_results', experiment)


def get_anf_folder(lvl,
                   impairment='none',
                   simulation_group='zeng_et_al_2005__sensitivity_test'):
    """
    Returns the path containing the ANF simulations for the given arguments

    @arg lvl:
        presentation level
    @arg impairment:
        type of impairment as a string
    @arg simulation_group:

    @return:
        path to ANF simulations
    """
    if impairment == 'none':
        return os.path.join(get_datafolder(),
                            '1_mdl_simulations',
                            f'{simulation_group}',
                            f'lvl_{lvl}_db_spl')
    else:
        return os.path.join(get_datafolder(),
                            '3_cs_simulations',
                            f'{impairment}',
                            f'{simulation_group}',
                            f'lvl_{lvl}_db_spl')


def get_spikeogram_folder(lvl,
                          mode='2d',
                          bin_width=1e-3,
                          impairment='none',
                          simulation_group='zeng_et_al_2005__sensitivity_test'):
    """
    Returns the path containing spikeograms for the given arguments

    @arg lvl:
        presentation level
    @arg mode:
        2d or 3d, i.e. all ANFs bundled or separate channels, LSR-, MSR- and HSR-fibers
    @arg bin_width:
        define the resolution; e.g. bin_width of 1ms results in 1kHz sampling rate
    @arg impairment:
        any impairment, default to 'none' or None
    @arg simulation_group:

    @return:
        path to spikeograms
    """
    if impairment is None or impairment == 'none':
        impairment = ''

    return os.path.join(get_datafolder(),
                        'cnn_data',
                        f'mode_{mode}',
                        f'bw_{str(bin_width)}',
                        impairment,
                        f'{simulation_group}',
                        f'lvl_{lvl}_db_spl')


def get_spikeograms_from_folder(lvl,
                                mode='2d',
                                bin_width=1e-3,
                                impairment='none',
                                simulation_group='zeng_et_al_2005__sensitivity_test'):
    """
    Returns spikeograms for the given arguments

    @arg lvl:
        presentation level
    @arg mode:
        2d or 3d, i.e. all ANFs bundled or separate channels, LSR-, MSR- and HSR-fibers
    @arg bin_width:
        define the resolution; e.g. bin_width of 1ms results in 1kHz sampling rate
    @arg impairment:
        any impairment, default to 'none' or None
    @arg simulation_group:

    @return:
        list of spikeogram files
    """
    folder = get_spikeogram_folder(lvl=lvl,
                                   mode=mode,
                                   bin_width=bin_width,
                                   impairment=impairment,
                                   simulation_group=simulation_group)

    return glob.glob(os.path.join(folder, '*.pt'))
