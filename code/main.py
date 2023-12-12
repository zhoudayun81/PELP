if __name__ == '__main__':
    import os, datetime, logging, multiprocessing, sys, itertools, torch
    from random import shuffle
    # import my own modules
    import config_reader, worker

    torch.manual_seed(333333333)

    upload_folder = 'upload'
    download_folder = 'download'
    input_folder = 'input'
    code_folder = 'code'
    output_folder = 'output'
    log_folder = 'log'
    model_folder = 'model'

    # The project working directory on Spartan should be: /data/gpfs/projects/punim1925/xxx
    CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_FOLDER))
    DOWNLOAD_DIR = os.path.join(PROJECT_DIR, download_folder)
    UPLOAD_DIR = os.path.join(PROJECT_DIR, upload_folder)
    # The code is by default that directory traversal doesn't apply.
    INPUT_DIR = config_reader.INPUT_DIR if config_reader.INPUT_DIR else os.path.join(UPLOAD_DIR, input_folder)
    OUTPUT_DIR = config_reader.OUTPUT_DIR if config_reader.OUTPUT_DIR else os.path.join(DOWNLOAD_DIR, output_folder)
    LOG_DIR = config_reader.LOG_DIR if config_reader.LOG_DIR else os.path.join(DOWNLOAD_DIR, log_folder)
    MODEL_DIR = config_reader.MODEL_DIR if config_reader.MODEL_DIR else os.path.join(DOWNLOAD_DIR, model_folder)
    INPUTFILES = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and not f.startswith('.')]
    THREADS = config_reader.THREADS

    # Create a logger for the main process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(LOG_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + '_main.log')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for section in config_reader.SECTIONS:
        for key in config_reader.CONFIG[section]:
            logger.info('Read from section %s, key %s, and its value = %s.' % (section, key, config_reader.CONFIG[section][key]))
    logger.info('CURRENT_FOLDER=%s'%CURRENT_FOLDER)
    logger.info('PROJECT_DIR=%s'%PROJECT_DIR)
    logger.info('DOWNLOAD_DIR=%s'%DOWNLOAD_DIR)
    logger.info('UPLOAD_DIR=%s'%UPLOAD_DIR)
    logger.info('INPUT_DIR=%s'%INPUT_DIR)
    logger.info('OUTPUT_DIR=%s'%OUTPUT_DIR)
    logger.info('LOG_DIR=%s'%LOG_DIR)
    logger.info('MODEL_DIR=%s'%MODEL_DIR)
    logger.info('Total %s number of files to experiment. INPUTFILES=%s'%(len(INPUTFILES), INPUTFILES))

    if config_reader.WARM_START and INPUTFILES and config_reader.ROWS_TO_FORECAST and config_reader.ROWS_TO_USE and config_reader.DROPOUT_PROBABILITY and config_reader.HIDDEN_SIZE and config_reader.LEARNING_RATE and config_reader.SUBSET_SIZE and config_reader.TEACHER_FORCING_RATIO and config_reader.EPOCHS:
        # If the work_on_file is False, then generate all configuration combinations, and each thread work on a bunch of configuration combinations.
        combo = list(itertools.product(config_reader.ROWS_TO_FORECAST, config_reader.ROWS_TO_USE, config_reader.DROPOUT_PROBABILITY, config_reader.HIDDEN_SIZE, config_reader.LEARNING_RATE, [config_reader.SUBSET_SIZE], config_reader.TEACHER_FORCING_RATIO, config_reader.LOSS_FUNCTION, config_reader.OPTIMIZER, INPUTFILES))
        shuffle(combo)
        param_groups = [combo[i::THREADS] for i in range(THREADS)]
        torch.multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            pool.starmap(worker.warm_start, [(params, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR) for params in param_groups])
    elif INPUTFILES and config_reader.ROWS_TO_FORECAST and config_reader.ROWS_TO_USE and config_reader.DROPOUT_PROBABILITY and config_reader.HIDDEN_SIZE and config_reader.LEARNING_RATE and config_reader.TRAINING_PERCENTAGE and config_reader.TEACHER_FORCING_RATIO and config_reader.EPOCHS:
        # If the work_on_file is False, then generate all configuration combinations, and each thread work on a bunch of configuration combinations.
        combo = list(itertools.product(config_reader.ROWS_TO_FORECAST, config_reader.ROWS_TO_USE, config_reader.DROPOUT_PROBABILITY, config_reader.HIDDEN_SIZE, config_reader.LEARNING_RATE, [config_reader.SUBSET_SIZE], config_reader.TEACHER_FORCING_RATIO, config_reader.LOSS_FUNCTION, config_reader.OPTIMIZER, INPUTFILES))
        shuffle(combo)
        param_groups = [combo[i::THREADS] for i in range(THREADS)]
        torch.multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            pool.starmap(worker.cold_start, [(params, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR) for params in param_groups])
    else:
        logger.error('Required parameter(s) or input files missing! Check the configuration file! Program exit!')
        sys.exit()
    
    # TODO: start measurement for all outputs, this can be done separately in the process_output.py though.