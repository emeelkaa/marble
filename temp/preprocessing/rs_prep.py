import os
import re
import pickle
import mne
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def get_patient_data(sub, scan, path_to_dataset, output_raw_eeg=True):
    patient = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path_set_file = os.path.join(path_to_dataset, 'eeg_set', f'{patient}_eeg.set')
    fMRI_path_difumo = os.path.join(path_to_dataset, 'difumo64', f'{patient}_difumo64_roi.pkl')
    new_fps = 200

    # Load and preprocess EEG
    raw = mne.io.read_raw_eeglab(eeg_path_set_file)
    original_fps = raw.info['sfreq']  # original_fps = 250
    if original_fps != new_fps:
        raw.resample(new_fps)
    if len(raw.ch_names) > 32:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG',
                          'CWL1', 'CWL2', 'CWL3', 'CWL4']
    else:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG']

    if output_raw_eeg == True:
        raw.drop_channels(vector_exclude)
        df_eeg = raw
    else:
        df_eeg = raw.to_data_frame()
        df_eeg = df_eeg.drop(vector_exclude, axis=1)

    # Load fMRI ROI time series (Difumo atlas) 
    df_fmri = pd.read_pickle(fMRI_path_difumo)
    roi_labels = df_fmri.columns.to_list()

    return df_eeg, df_fmri, roi_labels

def convert_to_tensor(data):
    tensors = [torch.tensor(arr) for arr in data]
    return torch.stack(tensors, dim=0)

def normalize_data(data):
    means = np.mean(data, axis=-1, keepdims=True)
    data = data - means
    transform_data = data / (
        np.quantile(np.abs(data), 0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
    return transform_data

def epoching_seq2one(eeg_raw, fmri, tmin, tmax, event_sync_name, crop=0, drop_fmri_frames=0):
    events, event_id = mne.events_from_annotations(eeg_raw)
    event_id = {event_sync_name: event_id[event_sync_name]}
    events = mne.pick_events(events, include=event_id[event_sync_name])

    if drop_fmri_frames != 0:
    # todo: if you are adapting this to a new dataset, make sure if needs dropping first several frames for mag stability
    # if there are multiple mr collection events within one scan, you could pick the first of every group of events
    # e.g.: events_1st = events[::30]
        events = events[drop_fmri_frames:]
    
    eeg_epoch = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                           preload=True, baseline=(None, None))

    eeg_epoch_data = eeg_epoch.get_data(units='uV')

    # get the indices for selected events for epoching
    select_ind = eeg_epoch.selection
    fmri_epoch = fmri[:, select_ind]

    eeg_epoch_sample = [np.array(sample) for sample in eeg_epoch_data]
    fmri_sample = [np.array(sample) for sample in fmri_epoch.T]  # here each time point is a sample, so transpose

    if crop > 0:
        eeg_epoch_sample = [sample[:, :crop] for sample in eeg_epoch_sample]  

    data = {"eeg": eeg_epoch_sample, "fmri": fmri_sample}

    return data, eeg_epoch  

def get_full_dataset(split_index_sheet, dataset_root, labels_roi=None, mri_sync_event='R149', 
                     TR=2.1, tmin=-16, tmax=0, crop=3200, save_input_tensor=True):
    df = pd.read_excel(io=split_index_sheet, sheet_name='fold1')
    scans = df['scan_name'].values
    train_ind = np.where(df['train'].values == 1)[0]
    val_ind = np.where(df['val'].values == 1)[0]
    test_ind = np.where(df['test'].values == 1)[0]

    # Data containers
    eeg_data = {"train": [], "val": [], "test": []}
    fmri_data = {"train": [], "val": [], "test": []}
    
    for scan in range(len(scans)):
        sub_ind, scan_ind = re.findall(r'\d+', scans[scan])
        print('Start preparing', scans[scan], '...')
        sub_ind, scan_ind = int(sub_ind), int(scan_ind)
        eeg_raw, df_fmri, labels_roi_full = get_patient_data(sub_ind, scan_ind, dataset_root)
        
        eeg_raw.load_data()
        eeg_raw.filter(l_freq=0.5, h_freq=None)
        
        if labels_roi is not None: 
            fmri_np = df_fmri[labels_roi].to_numpy().T
        else: 
            columns_to_exclude = ['global signal clean', 'global signal raw']
            fmri_np = df_fmri.drop(columns=columns_to_exclude, errors='ignore').to_numpy().T

        if len(fmri_np.shape) < 2:
            fmri_np = fmri_np[np.newaxis, ...]

        fs = 1 / TR
        nyquist = 0.5 * fs
        low = 0.15 / nyquist  
        b, a = butter(N=5, Wn=low, btype='low', analog=False)  
        fmri_np = filtfilt(b, a, fmri_np, axis=1)

        fmri_norm = normalize_data(fmri_np)

        data_epoch, eeg_info = epoching_seq2one(eeg_raw, fmri_norm, tmin, tmax, 
                                                mri_sync_event, crop=crop)

        if scan in train_ind:
            eeg_data["train"] += data_epoch["eeg"]
            fmri_data["train"] += data_epoch["fmri"]
        elif scan in val_ind:
            eeg_data["val"] += data_epoch["eeg"]
            fmri_data["val"] += data_epoch["fmri"]
        elif scan in test_ind:
            eeg_data["test"] += data_epoch["eeg"]
            fmri_data["test"] += data_epoch["fmri"]

    # convert to tensor
    eeg_tensors = {key: convert_to_tensor(eeg_data[key]) for key in eeg_data}
    fmri_tensors = {key: convert_to_tensor(fmri_data[key]).squeeze() for key in fmri_data}

    # Package data
    train_data = eeg_tensors["train"], fmri_tensors["train"]
    val_data = eeg_tensors["val"], fmri_tensors["val"]
    test_data = eeg_tensors["test"], fmri_tensors["test"]

    # Optional: save dataset here for saving time later
    if save_input_tensor:
        with open(f'testestest.pkl', 'wb') as f:
            pickle.dump([train_data, val_data, test_data], f)

    # Create TensorDatasets
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    test_dataset = TensorDataset(*test_data)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    '''
    split_index_sheet = 'neurobolt/splits/scan_split_example.xlsx'
    dataset_root = '../datasets/vu'
    #labels_roi = ["Cuneus", "Heschl’s gyrus", "Middle frontal gyrus anterior", "Precuneus anterior", "Thalamus", "Putamen"]

    train_dataset, val_dataset, test_dataset = get_full_dataset(split_index_sheet, dataset_root)
    print("Datasets successfully created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    '''
    out = get_patient_data(1, 1, '../datasets/audit')