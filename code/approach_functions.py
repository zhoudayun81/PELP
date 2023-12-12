#%%
import numpy as np
RANDOM_SEED = 333333333
SOS_TOKEN = 'SOS'
EOS_TOKEN = 'EOS'
EOT_TOKEN = 'EOT'
#UNK_TOKEN = 'IDK'
SOS_INDEX = 0
EOS_INDEX = 1
#UNK_INDEX = 2
#%%
# This method reads a file line by line, and stores each line as an element in the list 'lines'
def readLines(filepath, appendtoken=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() + f' {appendtoken}' for line in f.readlines()] if appendtoken else [line.strip() for line in f.readlines()]
    return lines       

# This method write lines into a file
def writeLinesToFile(lines, filepath, removetoken=None):
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace(removetoken, '').strip() + '\n') if removetoken else f.write(line + '\n')

#%%
# This function takes the traces and the number of history trace and the number of forecast trace as input
# and it outputs a list which contains the history and forecast pairs.
# e.g.: pairs = [ [history_trace0, forecast_trace0], [history_trace1, forecast_trace1], ... , [history_traceN, forecast_traceN] ]
# pairs[N][0] will get history_traceN, pairs[N][1] will get forecast_traceN
def prepareTracesPairs(traces, num_in, num_out):
    input = []
    output = []
    max_num_traces = len(traces) - (num_in + num_out) + 1
    for i in range(max_num_traces):
        tempx = ''
        tempy = ''
        for x in range(num_in):
            tempx += traces[i+x] + ' '
        input.append(tempx.strip())
        tempx = ''
        for y in range(num_out):
            tempy += traces[i+num_in+y] + ' '
        output.append(tempy.strip())
        tempy = ''
    return input, output

# Find the maximum length of a concatenated trace 
# (here I did not filter long traces, so the max length should be dependent on the longest trace either in the history part or the forecast part)
def find_max_trace_length(tensor1, tensor2):
    lengths1 = [len(tensor) for tensor in tensor1]
    lengths2 = [len(tensor) for tensor in tensor2]
    return max(max(lengths1), max(lengths2))

# Generate two directly-follows matrices (adjacency matrix) for two traces.
def compute_direct_follows(traces1, traces2):
    # Compute the set of unique activities in both arrays
    activities = sorted(set([a for t in traces1 + traces2 for a in t.split()]))
    # Initialize a matrix with zeros
    n = len(activities)
    matrix1 = [[0] * n for _ in range(n)]
    matrix2 = [[0] * n for _ in range(n)]
    # Compute the direct follows relationship for the first array
    for trace in traces1:
        events = trace.strip().split()
        for i in range(len(events) - 1):
            a = activities.index(events[i])
            b = activities.index(events[i+1])
            matrix1[a][b] += 1
    # Compute the direct follows relationship for the second array
    for trace in traces2:
        events = trace.strip().split()
        for i in range(len(events) - 1):
            a = activities.index(events[i])
            b = activities.index(events[i+1])
            matrix2[a][b] += 1
    # Return the two matrices
    return np.array(matrix1), np.array(matrix2)

def rmsd(ground_truth_mat, forecast_mat):
    # Root-mean-square deviation
    diff = forecast_mat - ground_truth_mat
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff)
    rmsd = np.sqrt(mean_squared_diff)
    return rmsd

def mae(ground_truth_mat, forecast_mat):
    # Mean Average Error
    diff = forecast_mat - ground_truth_mat
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    return mae 

# The (Normalized) Root Mean Square Error (NRMSE)
def NRMSE(ground_truth_mat, forecast_mat):
    diff = forecast_mat - ground_truth_mat
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    ground_truth_mean = np.mean(ground_truth_mat)
    nrmse = rmse / ground_truth_mean
    return nrmse

# The Symmetric Root Mean Square Percentage Error (sRMSPE)
def sRMSPE(ground_truth_mat, forecast_mat):
    diff = forecast_mat - ground_truth_mat
    sum = forecast_mat + ground_truth_mat
    squared_diff_sum = np.square(diff / sum)
    mean_squared_diff = np.mean(squared_diff_sum)
    srmspe = np.sqrt(mean_squared_diff)
    return srmspe

def measures(ground_truth_trace, forecast_trace, removetoken=None):
    if len(ground_truth_trace) == len(forecast_trace):
        if removetoken:
            ground_truth_trace = [trace.replace(removetoken, "").strip() for trace in ground_truth_trace]
            forecast_trace = [trace.replace(removetoken, "").strip() for trace in forecast_trace]
        gt_mat, f_mat = compute_direct_follows(ground_truth_trace, forecast_trace)
        rmsdv = rmsd(gt_mat, f_mat)
        maev = mae(gt_mat, f_mat)
        #srmspe = sRMSPE(gt_mat, f_mat)
    else:
        rmsdv = float('inf')
        maev = float('inf')
    return rmsdv, maev