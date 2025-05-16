"""
[2025-04-10]: added datetime (start_time, end_time, delta_time) to log the time taken for the process
[2025-04-10]: added sample_zie to show the number of samples in the dataset
[2025-05-12]: added execution time values
[2025-05-12]: added json_config.py / load_config
[2025-05-12]: added log_instance to get the log data in main.py
[2025-05-12]: added RESULTS_DIR to save the results
"""
### IMPORTS ###
import pickle
from torch.utils.data import DataLoader
from neural_network.HistoryDataset import CustomDataset
from transformers import AutoModel, AutoTokenizer
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from sklearn.model_selection import train_test_split
from preprocessing.log_to_history import Log
import torch
import random
import numpy as np
import sys
import time
from datetime import datetime
import pandas as pd

### IMPORTS (LOCAL) ###
from json_config import load_config
from functions import create_results_directory

### GLOBALS ###
SEED = 42
JSON_CONFIG = 'json_config.json'
RESULTS_DIR = 'results'
MULTIOUTPUT_DIR = 'multioutput'
MODELS_DIR = 'models'
LOG_HISTORY_DIR = 'log_history'

### FUNCTIONS ###

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

def train_fn(model, train_loader, optimizer, device, criterion):
    print("> Performing train_fn...")
    model.train()
    total_loss = 0
    for X_train_batch in train_loader:
        input_ids = X_train_batch['input_ids'].to(device)
        attention_mask = X_train_batch['attention_mask'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = 0
        for o, c, l in zip(output, criterion, X_train_batch['labels']):
            loss += criterion[c](o.to(device), X_train_batch['labels'][l].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_fn(model, data_loader, criterion, device):
    print("> Performing evaluate_fn...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            for o, c, l in zip(output, criterion, batch['labels']):
                total_loss += criterion[c](o.to(device), batch['labels'][l].to(device))
            total_loss = total_loss.item()
    return total_loss / len(data_loader)

def train_llm(model, train_data_loader, valid_data_loader, optimizer, EPOCHS, criterion):
        print("> Performing train_llm...")
        best_valid_loss = float("inf")
        early_stop_counter = 0
        patience = 5

        for epoch in range(EPOCHS):
            train_loss = train_fn(model, train_data_loader, optimizer, device, criterion)
            valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = model
                early_stop_counter = 0  # Reset early stopping counter
            else:
                early_stop_counter += 1

            print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {valid_loss:.4f}")
            if early_stop_counter >= patience:
                print("Validation loss hasn't improved for", patience, "epochs. Early stopping...")
                break
        return best_model

### MAIN ###
if __name__ == '__main__':

    print()
    print("*** PROGRAM START ***")
    print()

    start_time = datetime.now().replace(microsecond=0)
    print("Start process:", str(start_time))
    print()

    ### INPUTS ###
    if len(sys.argv) < 2:
        print("Error: Missing required argument 'event_log_name'. Please provide the log file name as the first argument (python -m main <event_log_name>).")
        sys.exit(1)
    csv_log = sys.argv[1]
    print('Log -->', csv_log)
    print()

    ### OUTPUTS ###
    print('> Creating results directory...')
    create_results_directory(RESULTS_DIR)
    print('Results directory:', RESULTS_DIR)
    create_results_directory(MULTIOUTPUT_DIR)
    print('Multioutput directory:', MULTIOUTPUT_DIR)
    create_results_directory(MODELS_DIR)
    print('Models directory:', MODELS_DIR)
    create_results_directory(MULTIOUTPUT_DIR + '/' + LOG_HISTORY_DIR)
    print('Log history directory:', LOG_HISTORY_DIR)

    ### CONFIGURATION ###
    config_dic = load_config(JSON_CONFIG)
    print("Configuration file loaded:", JSON_CONFIG)
    print(config_dic)
    print()

    ### SETTING PARAMETERS ###
    sample_zie = 3
    LOG_USE_COL = list(config_dic['LOG_USE_COL'])
    MAX_LEN = int(config_dic['MAX_LEN'])
    BATCH_SIZE = int(config_dic['BATCH_SIZE'])
    LEARNING_RATE = float(config_dic['LEARNING_RATE'])
    EPOCHS = int(config_dic['EPOCHS'])
    TYPE =  str(config_dic['TYPE'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device -->', device)
    print()


    ### PREPROCESSING LOG DATA ###
    print('> Preprocessing log data...')
    print("Input: columns to be used (empty = all):")
    print(LOG_USE_COL)
    log_instance = Log(csv_log, TYPE, LOG_USE_COL) # --> preprocessing/log_to_history.py
    event_log_df = log_instance.get_event_log()
    print()

    print('> Opening /log_history...')
    path_mo_lh = 'multioutput/log_history'
    with open(path_mo_lh + '/' + csv_log+'/'+csv_log+'_id2label_'+TYPE+'.pkl', 'rb') as f:
        id2label = pickle.load(f)

    with open(path_mo_lh + '/'+ csv_log+'/'+csv_log+'_label2id_'+TYPE+'.pkl', 'rb') as f:
        label2id = pickle.load(f)

    with open(path_mo_lh + '/' + csv_log+'/'+csv_log+'_train_'+TYPE+'.pkl', 'rb') as f:
        train = pickle.load(f)

    with open(path_mo_lh + '/' + csv_log+'/'+csv_log+'_label_train_'+TYPE+'.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open(path_mo_lh + '/' + csv_log+'/'+csv_log+'_suffix_train_'+TYPE+'.pkl', 'rb') as f:
        y_train_suffix = pickle.load(f)


    ### TRAINING DATASET ###
    print('> Performing train_test_split...')
    train_input, val_input = train_test_split(train, test_size=0.2, random_state=42)
    train_label = {}
    val_label = {}

    for key in y_train_suffix.keys():
        train_label[key], val_label[key] = train_test_split(y_train_suffix[key], test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', truncation_side='left')
    model = AutoModel.from_pretrained('prajjwal1/bert-medium')
    output_sizes = []

    for i in range(len(y_train_suffix)):
        output_sizes.append(len(id2label['activity']))
    print()

    print('> Extracting Training and Validation dataset...')
    print("Size of training input:", len(train_input))
    print("Size of validation input:", len(val_input))
    print("Size of training label:", len(train_label))
    print("Size of validation label:", len(val_label))
    print("MAX_LEN:", MAX_LEN)

    train_dataset = CustomDataset(train_input, train_label, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_input, val_label, tokenizer, MAX_LEN)
    # print("Type:", type(train_input)) # <class 'list'>
    print("Size of training input:", len(train_input))
    print(f"First {sample_zie} elements in train_input") # debug
    for i in range(sample_zie):
        print(f"[{i}]: {train_input[i]}")
    
    print()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print('> Training start...')
    # Initialize model
    model = BertMultiOutputClassificationHeads(model, output_sizes).to(device)
    criterion = {}

    for l in y_train_suffix:
        criterion[l] = torch.nn.CrossEntropyLoss()

    print('> Running optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    startTime = time.time()

    ### TRAINING LLM ###
    print('> Training LLM...')
    bert_model = train_llm(model, train_loader, val_loader, optimizer, EPOCHS, criterion)

    ### SAVING MODEL ###
    print('> Saving model...')
    path_model = 'models/'+csv_log+'_'+TYPE+'.pth'
    print('Path Model -->', path_model)
    torch.save(bert_model.state_dict(), path_model)
    
    ### EXECUTION TIME ###
    print('> Saving execution time...')
    executionTime = round((time.time() - startTime),3)
    hours, remainder = divmod(executionTime, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution Time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.3f} seconds")
    path_time = RESULTS_DIR + '/' + csv_log+'_'+TYPE+'_executionTime.txt'
    dic_res = {'event_log_name':csv_log, 'cases':event_log_df['case'].nunique(), 'event_log_length':len(event_log_df), 'execution_time_sec': executionTime}
    print('Path Execution Time and Dataframe informations -->', path_time)
    df_res = pd.DataFrame([dic_res])
    df_res.to_csv(path_time, index=False)

    end_time = datetime.now().replace(microsecond=0)
    delta_time = end_time - start_time

    print()
    print("End process:", end_time)
    print("Time to finish:", delta_time)
    print()

    print()
    print("*** PROGRAM END ***")
    print()