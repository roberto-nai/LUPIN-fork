"""
eval_model.py
[2025-05-12]: added json_config.py / load_config
[2025-05-14]: added DL values computed
"""
### IMPORTS ###
import sys
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from jellyfish._jellyfish import damerau_levenshtein_distance
from transformers import AutoModel, AutoTokenizer
import time
from datetime import datetime
import os
import pandas as pd

### IMPORTS (LOCAL) ###
from functions import create_results_directory

### GLOBALS ###
RESULTS_DIR = 'results'

### FUNCTIONS ###
def clean_sequence(sequence_str):
        sequence_list = sequence_str.split(' ')
        first_end_index = sequence_list.index(str(label2id['activity']['ENDactivity']))
        sequence_list = sequence_list[:first_end_index + 1]
        result_str = ' '.join(sequence_list)
        return result_str

def remove_word(sentence, word):
    words = sentence.split()
    words = [w for w in words if w != word]
    new_sentence = ' '.join(words)
    return new_sentence

### MAIN ###
if __name__ == '__main__':
    print()
    print("*** PROGRAM START ***")
    print()

    start_time = datetime.now().replace(microsecond=0)
    print("Start process:", str(start_time))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device-->', device)
    print()
    
    ### INPUTS ###
    if len(sys.argv) < 2:
        print("Error: Missing required argument 'event_log_name'. Please provide the log file name as the first argument (python -m eval_model <event_log_name>).")
        sys.exit(1)
    csv_log = sys.argv[1]
    print('Log -->', csv_log)
    print()

    ### OUTPUTS ###
    print('> Creating results directory...')
    print('Results directory:', RESULTS_DIR)
    create_results_directory(RESULTS_DIR)
    print()

    ### SETTING PARAMETERS ###
    TYPE = 'all'
    
    print('> Loading model...')
    with open('multioutput/log_history/'+csv_log+'/'+csv_log+'_test_'+TYPE+'.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('multioutput/log_history/'+csv_log+'/'+csv_log+'_label_test_'+TYPE+'.pkl', 'rb') as f:
        y_test = pickle.load(f)

    with open('multioutput/log_history/' + csv_log + '/' + csv_log + '_id2label_'+TYPE+'.pkl', 'rb') as f:
        id2label = pickle.load(f)

    with open('multioutput/log_history/' + csv_log + '/' + csv_log + '_label2id_'+TYPE+'.pkl', 'rb') as f:
        label2id = pickle.load(f)

    with open('multioutput/log_history/'+csv_log+'/'+csv_log+'_suffix_train_'+TYPE+'.pkl', 'rb') as f:
        y_train_suffix = pickle.load(f)

    with open('multioutput/log_history/'+csv_log+'/'+csv_log+'_suffix_test_'+TYPE+'.pkl', 'rb') as f:
        y_test_suffix = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', truncation_side='left')
    model = AutoModel.from_pretrained('prajjwal1/bert-medium')
    MAX_LEN = 512

    print('> Evaluating model...')
    test_dataset = CustomDataset(test, y_test_suffix, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    output_sizes = []

    dict_pred = {}
    dict_truth = {}
    for i in range(len(y_train_suffix)):
        output_sizes.append(len(id2label['activity']))

    model = BertMultiOutputClassificationHeads(model, output_sizes)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load('models/'+csv_log+'_'+TYPE+'.pth'))
    model = model.to(device)
    # Make sure to set the model in evaluation mode if you're not training it further
    model.eval()
    dict_pred = {}
    dict_truth = {}

    print('> Creating results...')
    list_dl_distance =[]
    path_dl = os.path.join(RESULTS_DIR, 'suffix_'+csv_log+'_'+TYPE+'.txt')
    print('Suffix file:', path_dl)
    file_dl = open(path_dl,'w')
    file_dl.write('pred,truth,dl_score\n')
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            #print(id2label['activity'])
            l_pred = []
            l_true = []
            for i in range(len(y_train_suffix)):
                pred = output[i].argmax(dim=1).cpu().numpy()
                l_pred.append(str(pred[0]))
                l_true.append(str(batch['labels'][i].item()))
            seq_pred = ' '.join(l_pred)
            seq_true = ' '.join(l_true)

            seq_pred = clean_sequence(seq_pred)
            seq_true = clean_sequence(seq_true)
            seq_pred = remove_word(seq_pred, str(label2id['activity']['ENDactivity']))
            seq_true = remove_word(seq_true, str(label2id['activity']['ENDactivity']))
            if seq_pred == '' and seq_true == '':
                seq_pred = 'end'
                seq_true = 'end'
            dl_distance = 1 - (damerau_levenshtein_distance(seq_pred, seq_true) / max(len(seq_pred), len(seq_true))) # DL distance
            file_dl.write(seq_pred+','+seq_true+','+str(dl_distance)+'\n')
            list_dl_distance.append(dl_distance)
    print(f"DL values computed -->  {len(list_dl_distance)}")
    print(f"DL average --> {np.mean(list_dl_distance):.3f}")
    print()

    print('> Saving DL...')
    path_dl = os.path.join(RESULTS_DIR, 'suffix_'+csv_log+'_'+TYPE+'_DL'+'.txt')
    print('DL file:', path_dl)
    with open(path_dl,'w') as file_dl:
        file_dl.write(f"DL --> {np.mean(list_dl_distance):.3f}\n")
    print()

    end_time = datetime.now().replace(microsecond=0)
    delta_time = end_time - start_time

    print()
    print("End process:", end_time)
    print("Time to finish:", delta_time)
    print()

    print()
    print("*** PROGRAM END ***")
    print()