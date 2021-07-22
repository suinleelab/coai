from matplotlib import colors

ED_DPATH_PROCESSED = '/homes/gws/erion/projects/pfrl/data/cwcf/'
ED_DPATH_RAW = '/projects/leelab2/ED_TRAUMA/'
ED_DATA_ALL = '../data/ed-trauma/all_years_2017.pickle'
ED_SURVEYPATH = '../data/ed-trauma/survey_time_report.csv'
ED_FNAME_FORMAT = 'TR All Patients and Data %s.xlsx'
ED_SHEET_FORMAT = 'All Patients and Data %d'
ED_SHEET_FORMAT_2 = 'All Patients and Data Part %d'
ED_ZIP_FILE = '../data/ed-trauma/newzips.csv'
ED_LATLNG_FILE = '../data/ed-trauma/census_zip_latlng.txt'
ED_PREHOSPITAL_VARS = '../data/ed-trauma/fieldvars.txt'
ED_INDIV_PATH = '../data/ed-trauma/trauma_indiv_times.csv'

EICU_DPATH = '/homes/gws/erion/projects/eicu_coai'
EICU_FIXED_SEED = 100
# EICU_IMPORTANCE_PATH = 'eicu_importance.npy'
EICU_SCORE_COSTS = {'qsofa':5,'aps':19, 'apacheiii':21, 'apacheiva':27}

OUTPATIENT_DPATH = '../data/nhanes/'

CDICT = {
    'exact': 'blue',
    'greedy': 'green',
    'rfe': 'orange',
    'increasingcost': 'cornflowerblue',
    'decreasingimportance': 'cornflowerblue',
    'pact':'purple',
    'rl':'red',
    'positivecontrol':'black'}

ED_LOWCOST = 0.0001
OUTPATIENT_LOWCOST = 0.001
ED_NAME = 'traumapaper'
ED_COSTTYPE = 'time'
ED_PACT_COST = 7.4
ED_SURVEY_BUDGET = 0.83

COMPLEXITY_MAX_ITER = 32

CWCF_TMPDIR = '/sdata/coai/'#'/data2/media/big/erion/coai/'

RUN_PATH = 'rebuttal_runs'

# Heatmap colors
cb_dict = {'red':   [[0.0,  0.9, 0.9],
                   [0.5,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.9, 0.9],
                   [0.5, 0.45, 0.45],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  1.0,1.0],
                   [0.5,  0.7, 0.7],
                   [1.0,  0.0, 0.0]]}
broken_cdict = {'red':   [[0.0,  0.8, 0.8],
                   [0.5,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.8, 0.8],
                   [0.5, 0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  0.0, 0.0]]}
# newcmp = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
cb_cmp = colors.LinearSegmentedColormap('testCmap', segmentdata=cb_dict, N=256)
cb_cmp.set_under('white')