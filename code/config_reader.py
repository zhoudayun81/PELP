import  os, sys, logging, configparser

config_file = 'config.ini'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE =  os.path.join(CURRENT_DIR, config_file)

if not os.path.exists(CONFIG_FILE): 
    logging.error("Config file does not exist! Program exit. Please check the file is located and named at: %s"%CONFIG_FILE)
    sys.exit()

CONFIG = configparser.ConfigParser(allow_no_value=True)
CONFIG.read(CONFIG_FILE)
SECTIONS = CONFIG.sections()
# --------- Universal Program Parameters ---------
INPUT_DIR = CONFIG['Program']['input_dir']
OUTPUT_DIR = CONFIG['Program']['output_dir']
ANA_DIR = CONFIG['Program']['analysis_dir']
LOG_DIR = CONFIG['Program']['log_dir']
MODEL_DIR = CONFIG['Program']['model_dir']
THREADS = int(CONFIG['Program']['threads'])
WORK_ON_FILE = CONFIG['Program'].getboolean('work_on_file')
# --------- End of Universal Program Parameters ---------

# --------- General Model Parameters ---------
#MODEL = CONFIG.get('Model','model').split()
OPTIMIZER = CONFIG.get('Model','optimzer').split()
LOSS_FUNCTION = CONFIG.get('Model','loss_function').split()
learning_rate = CONFIG.get('Model','learning_rate').split()
LEARNING_RATE = [float(x) for x in learning_rate]
training_percentage = CONFIG.get('Model','training_percentage').split()
TRAINING_PERCENTAGE = [float(x) for x in training_percentage]
EPOCHS = CONFIG['Model'].getint('epochs')
#BATCH_SIZE = CONFIG['Model'].getint('batch_size')
#INITIALISATION = CONFIG.get('Model','initialisation').split()
SATISFACTION = CONFIG['Model'].getfloat('satisfaction')
#TOLERANCE_COUNTER = CONFIG['Model'].getint('tolerance_counter')
# --------- End of General Model Parameters ---------

# --------- Model Specific Parameters ---------
teacher_forcing_ratio = CONFIG.get('Model-Specific','teacher_forcing_ratio').split()
TEACHER_FORCING_RATIO = [float(x) for x in teacher_forcing_ratio]
hidden_size = CONFIG.get('Model-Specific','hidden_size').split()
HIDDEN_SIZE = [int(x) for x in hidden_size]
dropout_probability = CONFIG.get('Model-Specific','dropout_probability').split()
DROPOUT_PROBABILITY = [float(x) for x in dropout_probability]
#topk = CONFIG.get('Model-Specific','topk').split()
#TOPK = [int(x) for x in topk]
# --------- End of Model Specific Parameters ---------

# --------- Approach Specific Parameters ---------
rows_to_use = CONFIG.get('Approach-Specific','rows_to_use').split()
ROWS_TO_USE = [int(x) for x in rows_to_use]
rows_to_forecast = CONFIG.get('Approach-Specific','rows_to_forecast').split()
ROWS_TO_FORECAST = [int(x) for x in rows_to_forecast]
SUBSET_SIZE = CONFIG['Approach-Specific'].getint('subset_size')
WARM_START = CONFIG['Approach-Specific'].getboolean('warm_start')
# --------- End of Approach Specific Parameters ---------