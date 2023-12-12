import random, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# import my own modules
import approach_functions as af

torch.manual_seed(af.RANDOM_SEED)
random.seed(af.RANDOM_SEED)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def indexesFromTrace(ds, trace):
    activities = trace.split()
    indexes = []
    for activity in activities:
        if activity != '':
            indexes.append(ds.activity2index[activity])
    return indexes

def tensorFromTraceNoEOS(ds, trace, device):
    indexes = indexesFromTrace(ds, trace)
    #TODO:questionable tensor shape
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromTrace(ds, trace, device):
    indexes = indexesFromTrace(ds, trace)
    indexes.append(af.EOS_INDEX)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def save_state_dict_only(state_dict, dictpath):
    torch.save(state_dict, dictpath)

def save_state_dict(state_dict, dictpath, content, textpath):
    torch.save(state_dict, dictpath)
    with open(textpath, 'w') as output:
        output.write(content)
    
class DS:
    def __init__(self):
        self.activity2index = {af.SOS_TOKEN:af.SOS_INDEX, af.EOS_TOKEN:af.EOS_INDEX}
        self.index2activity = {af.SOS_INDEX:af.SOS_TOKEN, af.EOS_INDEX:af.EOS_TOKEN}
        self.n_activity = 2  # Count SOS and EOS
    def addTrace(self, trace):
        for activity in trace.split():
            self.addActivity(activity)
    def addActivity(self, activity):
        if activity not in self.activity2index:
            self.activity2index[activity] = self.n_activity
            self.index2activity[self.n_activity] = activity
            self.n_activity += 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, input, hidden):
        #TODO: tensor shape questionable
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

# class combining the encoder and decoder for training
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, criterion, max_length):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion # there may be a need of change when the criterion changes to different types.
        self.max_length = max_length
    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio):
        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=DEVICE)
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[af.SOS_INDEX]], device=DEVICE)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == af.EOS_INDEX:
                    break     
        return loss / target_length

# test function
def test(tracetotal, model, trace, rows_to_forecast):
    pattern = r'(.*? {})'.format(re.escape(af.EOT_TOKEN)) # This will find all the traces with EOT token at the end, the returned traces will also include the "EOT" token. 
    # move {} outside the () to the end, and the returned traces will not have "EOT" token.
    individual_traces = re.findall(pattern, trace)
    balance = model.max_length - len(' '.join(individual_traces[1:]).strip().split())
    with torch.no_grad():
        input_tensor = tensorFromTraceNoEOS(tracetotal, trace, DEVICE)
        input_length = input_tensor.size(0)
        encoder_hidden = model.encoder.initHidden()
        encoder_outputs = torch.zeros(model.max_length, model.encoder.hidden_size, device=DEVICE)
        for ei in range(input_length):
            encoder_output, encoder_hidden = model.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        #TODO: the question here is to understand whether [[af.SOS_INDEX]] is right, or [af.SOS_INDEX] is right
        decoder_input = torch.tensor([[af.SOS_INDEX]], device=DEVICE)
        decoder_hidden = encoder_hidden
        decoded_activities = []
        decoder_attentions = torch.zeros(model.max_length, model.max_length)
        count = 0
        for di in range(balance):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == af.EOS_INDEX or count >= rows_to_forecast:
                break
            else:
                decoded_activities.append(tracetotal.index2activity[topi.item()])
                if tracetotal.index2activity[topi.item()] == af.EOT_TOKEN:
                    count += 1
            decoder_input = topi.squeeze().detach()
        return decoded_activities

# define loss function
def loss_criterion(loss_function_name):
    #by default, use mean absolute error (MAE)
    criterion = nn.L1Loss()
    if loss_function_name=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name=='MSELoss':
        criterion = nn.MSELoss()
    elif loss_function_name=='CTCLoss':
        criterion = nn.CTCLoss() # used differently
    elif loss_function_name=='NLLLoss':
        criterion = nn.NLLLoss()
    elif loss_function_name=='PoissonNLLLoss':
        criterion = nn.PoissonNLLLoss()
    elif loss_function_name=='GaussianNLLLoss':
        criterion = nn.GaussianNLLLoss() # used differently
    elif loss_function_name=='KLDivLoss':
        criterion = nn.KLDivLoss() 
    elif loss_function_name=='BCELoss':
        criterion = nn.BCELoss()
    elif loss_function_name=='BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function_name=='MarginRankingLoss':
        criterion = nn.MarginRankingLoss() # used differently
    elif loss_function_name=='HingeEmbeddingLoss':
        criterion = nn.HingeEmbeddingLoss()
    elif loss_function_name=='MultiLabelMarginLoss':
        criterion = nn.MultiLabelMarginLoss()
    elif loss_function_name=='HuberLoss':
        criterion = nn.HuberLoss()
    elif loss_function_name=='SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    elif loss_function_name=='SoftMarginLoss':
        criterion = nn.SoftMarginLoss()
    elif loss_function_name=='MultiLabelSoftMarginLoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif loss_function_name=='CosineEmbeddingLoss':
        criterion = nn.CosineEmbeddingLoss()
    elif loss_function_name=='MultiMarginLoss':
        criterion = nn.MultiMarginLoss()
    elif loss_function_name=='TripletMarginLoss':
        criterion = nn.TripletMarginLoss() # used differently
    elif loss_function_name=='TripletMarginWithDistanceLoss':
        criterion = nn.TripletMarginWithDistanceLoss() # used differently
    return criterion

# define optimizer
def optimizer_function(optimizer_name, model, learning_rate):
    # by default, use SGD
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name=='Adadelta':
        optimiser = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Adadelta':
        optimiser = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Adam':
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='AdamW':
        optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name=='SparseAdam':
        optimiser = optim.SparseAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='ASGD':
        optimiser = optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_name=='LBFGS':
        optimiser = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_name=='NAdam':
        optimiser = optim.NAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='RAdam':
        optimiser = optim.RAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='RMSprop':
        optimiser = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Rprop':
        optimiser = optim.Rprop(model.parameters(), lr=learning_rate)
    return optimiser