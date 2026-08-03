from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.davis_dir = ''
    settings.result_plot_path = ''
    settings.results_path = ''    # Where to store tracking results56
    settings.segmentation_path = ''
    settings.network_path = ''  # Where tracking networks are stored.
    settings.prj_dir = ''
    settings.save_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.lasot_extension_subset_path=''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tnl2k_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.youtubevos_dir = ''

    settings.youtubevos_dir = ''
    return settings

#/media/ducktom/DATA/SOT/MODEL/O/MCITrack2