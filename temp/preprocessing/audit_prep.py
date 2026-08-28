import os
import re
import pickle
import mne
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import numpy as np
from scipy.signal import butter, filtfilt

def normalize_data(data):
    means = np.mean(data, axis=-1, keepdims=True)
    data = data - means
    transform_data = data / (
        np.quantile(np.abs(data), 0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
    return transform_data

def convert_to_tensor(data):
    tensors = [torch.tensor(arr) for arr in data]
    return torch.stack(tensors, dim=0)

def get_patient_data(sub, scan, root):
    patient = f'sub_00{sub:02d}-mr_00{scan:02d}'
    eeg_set_path = os.path.join(root, 'EEG', f"{patient}-ect_echo1_EEG_pp.set")
    fmri_difumo_path = os.path.join(root, 'FMRI', f"{patient}_difumo_roi.pkl")
    new_fps = 200

    raw = mne.io.read_raw_eeglab(eeg_set_path)

    original_fps = raw.info['sfreq']  # original_fps = 250
    if original_fps != new_fps:
        raw.resample(new_fps)

    vector_exclude = ['FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'ECG']
    raw.drop_channels(vector_exclude)

    df_fmri = pd.read_pickle(fmri_difumo_path)
    roi_labels = df_fmri.columns.to_list()
    return raw, df_fmri, roi_labels

def epoching(eeg, fmri, tmin, tmax, event_sync_name, crop=0, drop_fmri_frames=0):
    events, event_id = mne.events_from_annotations(eeg)
    event_id = {event_sync_name: event_id[event_sync_name]}
    events = mne.pick_events(events, include=event_id[event_sync_name])
    
    events = events[::30]

    if drop_fmri_frames != 0:
        events = events[drop_fmri_frames:]

    eeg_epoch = mne.Epochs(eeg, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                           preload=True, baseline=(None, None))
    
    eeg_epoch_data = eeg_epoch.get_data(units='uV')

    select_ind = eeg_epoch.selection
    fmri_epoch = fmri[:, select_ind]

    eeg_epoch_sample = [np.array(sample) for sample in eeg_epoch_data]
    fmri_sample      = [np.array(sample) for sample in fmri_epoch.T]

    if crop > 0:
        eeg_epoch_sample = [sample[:, :crop] for sample in eeg_epoch_sample]

    data = {"eeg": eeg_epoch_sample, "fmri": fmri_sample}
    return data

def get_dataset(split_index_sheet, root, mri_sync_event='R128', TR=2.1, 
                tmin=-16, tmax=0, crop=3200, save_input_tensor=True):
    df = pd.read_excel(io=split_index_sheet, sheet_name='fold1')
    scans = df['scan_name'].values
    train_ind = np.where(df['train'].values == 1)[0]
    val_ind = np.where(df['val'].values == 1)[0]
    test_ind = np.where(df['test'].values == 1)[0]

    # Data containers
    splits = {"train": [], "val": [], "test": []}
    
    for scan in range(len(scans)):
        sub_ind, scan_ind = re.findall(r'\d+', scans[scan])
        print('Start preparing', scans[scan], '...')
        sub_ind, scan_ind = int(sub_ind), int(scan_ind)
        eeg_raw, df_fmri, labels_roi_full = get_patient_data(sub_ind, scan_ind, root)

        eeg_raw.load_data()
        eeg_raw.filter(l_freq=0.5, h_freq=None)

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

        data_epoch = epoching(eeg_raw, fmri_norm, tmin, tmax, 
                                mri_sync_event, crop=crop, drop_fmri_frames=7)
        
        eeg_tensor  = torch.stack([torch.tensor(e, dtype=torch.float32) 
                                   for e in data_epoch["eeg"]], dim=0)   
        fmri_tensor = torch.stack([torch.tensor(f, dtype=torch.float32) 
                                   for f in data_epoch["fmri"]], dim=0)
        print(f"  -> EEG tensor: {eeg_tensor.shape}, fMRI tensor: {fmri_tensor.shape}")

        if scan in train_ind:
            splits["train"].append((eeg_tensor, fmri_tensor))
        elif scan in val_ind:
            splits["val"].append((eeg_tensor, fmri_tensor))
        elif scan in test_ind:
            splits["test"].append((eeg_tensor, fmri_tensor))
    
    if save_input_tensor:
        with open('audit2.pkl', 'wb') as f:
            pickle.dump(splits, f)
        print("Saved dataset to audit2.pkl")

    # Create TensorDatasets
    train_dataset = splits["train"]
    val_dataset   = splits["val"]
    test_dataset  = splits["test"]

    return train_dataset, val_dataset, test_dataset


root = "../../datasets/audit/"
print(os.listdir(root))

ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 
            'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'OZ', 'TP9', 'TP10', 'POZ']
split_index_sheet = 'splits/audit_2.xlsx'
get_dataset(split_index_sheet, root)