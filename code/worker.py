#%%
import os, logging, multiprocessing, datetime, re
# import my own modules
import config_reader
import seq2seq_RNNattention as s2s
import approach_functions as af

def warm_start(params, inputdir, outputdir, log_path, model_dir):
    # Create a logger for this worker process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_path, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + f'_{multiprocessing.current_process().name}.log')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for param in params:
        rows_to_forecast = param[0]
        rows_to_use = param[1]
        dropout_p = param[2]
        hidden_size = param[3]
        learning_rate = param[4]
        subset_size = param[5]
        teacher_forcing_ratio = param[6]
        loss_function = param[7]
        optimizer = param[8]
        filename = param[9]
        logger.info(f'loss_function:{loss_function}, hidden_size:{hidden_size}, dropout_p:{dropout_p}, optimizer:{optimizer}, learning_rate:{learning_rate}, teacher_forcing_ratio:{teacher_forcing_ratio}, subset_size:{subset_size}, rows_to_forecast:{rows_to_forecast}, rows_to_use:{rows_to_use}')
        # ----------------- Data Preprocessing -----------------
        trace_file_path = os.path.join(inputdir, filename)
        traces = af.readLines(trace_file_path, af.EOT_TOKEN)
        logger.info(f"Read {filename} {len(traces)} trace lines")
        #subsets = af.slice_string_list(traces, subset_size)
        quotient = (len(traces) - (rows_to_use + rows_to_forecast) + 1) // subset_size
        slicing_index = (subset_size-1) * quotient + rows_to_use + rows_to_forecast - 1
        logger.info(f'Slicing index is: {slicing_index}; Quotient is: {quotient} (round up for the subset size {subset_size} defined.)')
        training_traces = traces[:slicing_index]
        testing_traces = traces[slicing_index:]
        if rows_to_forecast > rows_to_use or (rows_to_use + rows_to_forecast) >= len(training_traces):
            logger.info(f'skipped {rows_to_use} -> {rows_to_forecast}.')
            continue
        logger.info(f'Start processing, {rows_to_use} -> {rows_to_forecast}')
        training_input, training_ground_truth = af.prepareTracesPairs(training_traces, rows_to_use, rows_to_forecast)
        tracetotal = s2s.DS()
        training_input_tensors = []
        training_ground_truth_tensors = []
        for hi,fo in zip(training_input, training_ground_truth):
            tracetotal.addTrace(hi)
            # it is not necessary to append EOS tokem for training input trace
            training_input_tensors.extend([s2s.tensorFromTraceNoEOS(tracetotal, hi, s2s.DEVICE)])
            tracetotal.addTrace(fo)
            # it is necessary to append the EOS token for training target trace to allow the model to stop generating output
            training_ground_truth_tensors.extend([s2s.tensorFromTrace(tracetotal, fo, s2s.DEVICE)])
        logger.info('Number of activities in training traces (including SOS, EOS and EOT token): %s'%tracetotal.n_activity)
        # split the data input subset number of equal sets
        training_input_tensor_sets = [training_input_tensors[i:i+quotient] for i in range(0, len(training_input_tensors), quotient)]
        training_ground_truth_tensor_sets = [training_ground_truth_tensors[i:i+quotient] for i in range(0, len(training_ground_truth_tensors), quotient)]      
        # ----------------- End of Data Preprocessing -----------------\
        max_length = af.find_max_trace_length(training_input_tensors, training_ground_truth_tensors)
        logger.info('Training max trace length: %s'%max_length)
        # ----------------- Training -----------------
        # Iterate through subsets and train
        for start_idx in range(len(training_input_tensor_sets)):
            # ----------------- Model Initialisation -----------------
            encoder = s2s.EncoderRNN(tracetotal.n_activity, hidden_size).to(s2s.DEVICE)
            attn_decoder = s2s.AttnDecoderRNN(hidden_size, tracetotal.n_activity, dropout_p, max_length).to(s2s.DEVICE)
            criterion = s2s.loss_criterion(loss_function)
            logger.debug('Loss Criterion defined.')
            model = s2s.Seq2Seq(encoder, attn_decoder, criterion, max_length)
            logger.debug('Model defined.')
            modeloptimizer = s2s.optimizer_function(optimizer, model, learning_rate)
            logger.debug('Optimizer defined.')
            # ----------------- End of Model Initialisation -----------------
            for end_idx in range(start_idx + 1, len(training_input_tensor_sets) + 1):
                training_input_tensor_sub = training_input_tensor_sets[start_idx:end_idx][0]
                training_ground_truth_tensor_sub = training_ground_truth_tensor_sets[start_idx:end_idx][0]
                best_val_loss = float('inf')
                best_state_dict = None
                best_epoc = 0
                universal_name = f'{os.path.basename(filename)[:-4]}_{subset_size}_{rows_to_use}-{rows_to_forecast}_{loss_function}_{hidden_size}_{dropout_p}_{optimizer}_{learning_rate}_{teacher_forcing_ratio}_{config_reader.EPOCHS}_{start_idx}..{end_idx}'
                model.train()
                for e in range(config_reader.EPOCHS):
                    running_loss = 0
                    #len(input_tensors) should be something from dataloader
                    for row in range(len(training_input_tensor_sub)):
                        modeloptimizer.zero_grad()
                        loss = model(training_input_tensor_sub[row], training_ground_truth_tensor_sub[row], teacher_forcing_ratio)
                        loss.backward()
                        modeloptimizer.step()
                        running_loss += loss.item()
                    logger.info(f'Epoch {e+1}: Loss: {running_loss}; Loss devided by total no. of pairs: {running_loss/len(training_input_tensor_sub)};')
                    # save the current model if validation loss is the best so far
                    if running_loss < best_val_loss:
                        best_val_loss = running_loss
                        best_state_dict = model.state_dict()
                        best_epoc = e+1
                        hyperparameters = f'The best settings for {filename} is:\nrows_to_use: {rows_to_use}\nrows_to_forecast: {rows_to_forecast}\nhidden_size: {hidden_size}\nepoch: {best_epoc} out of {config_reader.EPOCHS}\ndropout_p: {dropout_p}\nteacher_forcing_ratio: {teacher_forcing_ratio}\nlearning_rate: {learning_rate}\noptimizer: {optimizer}\nloss_function: {loss_function}\nsubset_size: {subset_size}\nmodel: EncoderRNN_AttnDecoderRNN\nloss: {best_val_loss}\nsubset: {start_idx}\nnumber_of_susequent_set: {end_idx}'
                        state_dict_path = os.path.join(model_dir,universal_name+'_best_state_dict.pt')
                        text_path = os.path.join(model_dir, universal_name+f'_hyperparams.txt')
                        s2s.save_state_dict(best_state_dict, state_dict_path, hyperparameters, text_path)
                        logger.info(f'The best model saved at: {model_dir}\{universal_name}') 
                logger.info('Training done.')
                # if implements early stopping mechanism, control should be here.
                # ----------------- End of Training -----------------
                # After training, test the training dataset accuracy using the best training params
                model.load_state_dict(best_state_dict)
                model.eval()
                # with torch.no_grad(): is called in test method
                # ----------------- Testing -----------------
                testing_pred = []
                # for simplifying code logic, get the last n rows from the training traces as the first forecast input, so the first N(rows_to_use) traces in testing_pred are not predictions.
                testing_pred.extend(training_traces[-rows_to_use:])
                last_forecast_seed = ' '.join(testing_pred)
                begin = len(training_input_tensor_sub) + rows_to_forecast + rows_to_use - 1
                while (len(testing_pred) - rows_to_use) < len(traces[begin:]):
                    forecast_activity_trace = s2s.test(tracetotal, model, last_forecast_seed, rows_to_forecast)
                    forecast_results = ' '.join(forecast_activity_trace)
                    pattern = r'(.*? {})'.format(re.escape(af.EOT_TOKEN)) # This will find all the traces with EOT token at the end, the returned traces will also include the "EOT" token. move {} outside the () to the end, and the returned traces will not have "EOT" token.
                    individual_traces = re.findall(pattern, forecast_results)
                    if len(individual_traces) < rows_to_forecast:
                        # break the while loop if the prediction is not satisfactory for 10 trial (otherwise the code may be prone to forever loop)
                        logger.error(f'Forecast is unusual, break from the loop! Forecast results: {forecast_results}.')
                        break
                    for t in individual_traces:
                        testing_pred.append(t.strip())
                    last_forecast_seed = ' '.join(testing_pred[-rows_to_use:])
                # remove the appended historical traces added at the beginning
                testing_pred = testing_pred[rows_to_use:]
                # remove extra forecasts
                if len(testing_pred)>len(testing_traces):
                    testing_pred = testing_pred[:len(testing_traces)]
                predicted_trace_path = os.path.join(outputdir,universal_name + f'_seq2seq.txt')
                af.writeLinesToFile(testing_pred, predicted_trace_path, af.EOT_TOKEN)
                logger.info(f'filename:{filename[:-4]}__rows_to_use:{rows_to_use}-rows_to_forecast:{rows_to_forecast}__hidden_size:{hidden_size}__best_epoc:{best_epoc}__dropout_p:{dropout_p}__teacher_forcing_ratio:{teacher_forcing_ratio}__learning_rate:{learning_rate}_optimizer:{optimizer}__loss_function:{loss_function}__subset_size:{subset_size}. Prediction generated at:{predicted_trace_path}')
                # ----------------- End of Testing -----------------

def cold_start(params, inputdir, outputdir, log_path, model_dir):
    # Create a logger for this worker process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_path, datetime.datetime.now().strftime('%Y-%m-%d_%H=%M=%S') + f'_{multiprocessing.current_process().name}.log')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for param in params:
        rows_to_forecast = param[0]
        rows_to_use = param[1]
        dropout_p = param[2]
        hidden_size = param[3]
        learning_rate = param[4]
        subset_size = param[5]
        teacher_forcing_ratio = param[6]
        loss_function = param[7]
        optimizer = param[8]
        filename = param[9]
        logger.info(f'loss_function:{loss_function}, hidden_size:{hidden_size}, dropout_p:{dropout_p}, optimizer:{optimizer}, learning_rate:{learning_rate}, teacher_forcing_ratio:{teacher_forcing_ratio}, subset_size:{subset_size}, rows_to_forecast:{rows_to_forecast}, rows_to_use:{rows_to_use}')
        # ----------------- Data Preprocessing -----------------
        trace_file_path = os.path.join(inputdir, filename)
        traces = af.readLines(trace_file_path, af.EOT_TOKEN)
        logger.info(f"Read {filename} {len(traces)} trace lines")
        #subsets = af.slice_string_list(traces, subset_size)
        quotient = (len(traces) - (rows_to_use + rows_to_forecast) + 1) // subset_size
        slicing_index = (subset_size-1) * quotient + rows_to_use + rows_to_forecast - 1
        logger.info(f'Slicing index is: {slicing_index} (round up for the subset size {subset_size} defined.)')
        training_traces = traces[:slicing_index]
        testing_traces = traces[slicing_index:]
        if rows_to_forecast > rows_to_use or (rows_to_use + rows_to_forecast) >= len(training_traces):
            logger.info(f'skipped {rows_to_use} -> {rows_to_forecast}.')
            continue
        logger.info(f'Start processing, {rows_to_use} -> {rows_to_forecast}')
        training_input, training_ground_truth = af.prepareTracesPairs(training_traces, rows_to_use, rows_to_forecast)
        tracetotal = s2s.DS()
        training_input_tensors = []
        training_ground_truth_tensors = []
        for hi,fo in zip(training_input, training_ground_truth):
            tracetotal.addTrace(hi)
            # it is not necessary to append EOS tokem for training input trace
            training_input_tensors.extend([s2s.tensorFromTraceNoEOS(tracetotal, hi, s2s.DEVICE)])
            tracetotal.addTrace(fo)
            # it is necessary to append the EOS token for training target trace to allow the model to stop generating output
            training_ground_truth_tensors.extend([s2s.tensorFromTrace(tracetotal, fo, s2s.DEVICE)])
        logger.info('Number of activities in training traces (including SOS, EOS and EOT token): %s'%tracetotal.n_activity)
        # split the data input subset number of equal sets
        training_input_tensor_sets = [training_input_tensors[i:i+quotient] for i in range(0, len(training_input_tensors), quotient)]
        training_ground_truth_tensor_sets = [training_ground_truth_tensors[i:i+quotient] for i in range(0, len(training_ground_truth_tensors), quotient)]
        slen = len(training_input_tensor_sets[0])
        # ----------------- End of Data Preprocessing -----------------
        max_length = af.find_max_trace_length(training_input_tensors, training_ground_truth_tensors)
        logger.info('Training max trace length: %s'%max_length)
        # ----------------- Training -----------------
        # Iterate through subsets and train
        training_input_tensor_sets = []
        training_ground_truth_tensor_sets = []
        for i in range(0, subset_size):
            for j in range(i + 1, subset_size):
                training_input_tensor_sets.append(training_input_tensor_sets[slen*i:slen*j])
                training_ground_truth_tensor_sets.append(training_ground_truth_tensor_sets[slen*i:slen*j])
        for m in range(len(training_input_tensor_sets)):
            # ----------------- Model Initialisation -----------------
            encoder = s2s.EncoderRNN(tracetotal.n_activity, hidden_size).to(s2s.DEVICE)
            attn_decoder = s2s.AttnDecoderRNN(hidden_size, tracetotal.n_activity, dropout_p, max_length).to(s2s.DEVICE)
            criterion = s2s.loss_criterion(loss_function)
            logger.debug('Loss Criterion defined.')
            model = s2s.Seq2Seq(encoder, attn_decoder, criterion, max_length)
            logger.debug('Model defined.')
            modeloptimizer = s2s.optimizer_function(optimizer, model, learning_rate)
            logger.debug('Optimizer defined.')
            # ----------------- End of Model Initialisation -----------------
            best_val_loss = float('inf')
            best_state_dict = None
            best_epoc = 0
            training_input_tensor_sub = training_input_tensor_sets[m]
            training_ground_truth_tensor_sub = training_ground_truth_tensor_sets[m]
            universal_name = f'{os.path.basename(filename)[:-4]}_{subset_size}_{rows_to_use}-{rows_to_forecast}_{loss_function}_{hidden_size}_{dropout_p}_{optimizer}_{learning_rate}_{teacher_forcing_ratio}_{config_reader.EPOCHS}_{m}'
            model.train()
            for e in range(config_reader.EPOCHS):
                running_loss = 0
                #len(input_tensors) should be something from dataloader
                for row in range(len(training_input_tensor_sub)):
                    modeloptimizer.zero_grad()
                    loss = model(training_input_tensor_sub[row], training_ground_truth_tensor_sub[row], teacher_forcing_ratio)
                    loss.backward()
                    modeloptimizer.step()
                    running_loss += loss.item()
                logger.info(f'Epoch {e+1}: Loss: {running_loss}; Loss devided by total no. of pairs: {running_loss/len(training_input_tensor_sub)};')
                # save the current model if validation loss is the best so far
                if running_loss < best_val_loss:
                    best_val_loss = running_loss
                    best_state_dict = model.state_dict()
                    best_epoc = e+1
                    hyperparameters = f'The best settings for {filename} is:\nrows_to_use: {rows_to_use}\nrows_to_forecast: {rows_to_forecast}\nhidden_size: {hidden_size}\nepoch: {best_epoc} out of {config_reader.EPOCHS}\ndropout_p: {dropout_p}\nteacher_forcing_ratio: {teacher_forcing_ratio}\nlearning_rate: {learning_rate}\noptimizer: {optimizer}\nloss_function: {loss_function}\nsubset_size: {subset_size}\nmodel: EncoderRNN_AttnDecoderRNN\nloss: {best_val_loss}\nsubset: {m}\n'
                    state_dict_path = os.path.join(model_dir,universal_name+'_best_state_dict.pt')
                    text_path = os.path.join(model_dir, universal_name+f'_hyperparams.txt')
                    s2s.save_state_dict(best_state_dict, state_dict_path, hyperparameters, text_path)
                    logger.info(f'The best model saved at: {model_dir}\{universal_name}') 
            logger.info('Training done.')
            # if implements early stopping mechanism, control should be here.
            # ----------------- End of Training -----------------
            # After training, test the training dataset accuracy using the best training params
            model.load_state_dict(best_state_dict)
            model.eval()
            # with torch.no_grad(): is called in test method
            # ----------------- Testing -----------------
            testing_pred = []
            # for simplifying code logic, get the last n rows from the training traces as the first forecast input, so the first N(rows_to_use) traces in testing_pred are not predictions.
            testing_pred.extend(training_traces[-rows_to_use:])
            last_forecast_seed = ' '.join(testing_pred)
            begin = len(training_input_tensor_sub) + rows_to_forecast + rows_to_use - 1
            while (len(testing_pred) - rows_to_use) < len(traces[begin:]):
                forecast_activity_trace = s2s.test(tracetotal, model, last_forecast_seed, rows_to_forecast)
                forecast_results = ' '.join(forecast_activity_trace)
                pattern = r'(.*? {})'.format(re.escape(af.EOT_TOKEN)) # This will find all the traces with EOT token at the end, the returned traces will also include the "EOT" token. move {} outside the () to the end, and the returned traces will not have "EOT" token.
                individual_traces = re.findall(pattern, forecast_results)
                if len(individual_traces) < rows_to_forecast:
                    # break the while loop if the prediction is not satisfactory for 10 trial (otherwise the code may be prone to forever loop)
                    logger.error(f'Forecast is unusual, break from the loop! Forecast results: {forecast_results}.')
                    break
                for t in individual_traces:
                    testing_pred.append(t.strip())
                last_forecast_seed = ' '.join(testing_pred[-rows_to_use:])
            # remove the appended historical traces added at the beginning
            testing_pred = testing_pred[rows_to_use:]
            # remove extra forecasts
            if len(testing_pred)>len(testing_traces):
                testing_pred = testing_pred[:len(testing_traces)]
            predicted_trace_path = os.path.join(outputdir,universal_name + f'_seq2seq.txt')
            af.writeLinesToFile(testing_pred, predicted_trace_path, af.EOT_TOKEN)
            logger.info(f'filename:{filename[:-4]}__rows_to_use:{rows_to_use}-rows_to_forecast:{rows_to_forecast}__hidden_size:{hidden_size}__best_epoc:{best_epoc}__dropout_p:{dropout_p}__teacher_forcing_ratio:{teacher_forcing_ratio}__learning_rate:{learning_rate}_optimizer:{optimizer}__loss_function:{loss_function}__subset_size:{subset_size}. Prediction generated at:{predicted_trace_path}')
            # ----------------- End of Testing -----------------
#%% Test on no threads
if __name__ == '__main__':
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
    #INPUTFILES = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and not f.startswith('.')]
    INPUTFILES = os.path.join(INPUT_DIR, '1Sepsis_traces.txt')

    params = [[config_reader.ROWS_TO_FORECAST[0], config_reader.ROWS_TO_USE[0], config_reader.DROPOUT_PROBABILITY[0], config_reader.HIDDEN_SIZE[0], config_reader.SUBSET_SIZE, config_reader.SUBSET_SIZE, config_reader.TEACHER_FORCING_RATIO[0], config_reader.LOSS_FUNCTION[0], config_reader.OPTIMIZER[0], INPUTFILES]]
    cold_start(params, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR)
