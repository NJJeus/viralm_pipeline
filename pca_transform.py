import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
import os
import pandas as pd
import random
import pickle
import argparse
import sys
import tqdm
import json

parser = argparse.ArgumentParser(description='Process input files and write to output file or stdout.')

# Unlimited input files (nargs='*' for zero or more, '+' for one or more)
parser.add_argument('input_files', nargs='*', help='Input file(s)')
parser.add_argument('-p', '--scaler_pca', type=str, default=sys.stdin.buffer,
        help='set of (scaler, pca) as pkl file or stdin.buffer')

# Optional output file, defaults to stdout if not provided
parser.add_argument('-o', '--output',
                    type=str,
                    default=sys.stdout.buffer,
                    help='Output file (default: stdout)')

# New optional argument for CSV file and N value
parser.add_argument('--csv',
                    type=str,
                    help='CSV file containing Sample_N columns and file basenames')
parser.add_argument('--sample_n',
                    type=int,
                    help='N value to select files from Sample_N column in CSV')

args = parser.parse_args()

def pca_transform(scaler_pca, X, file_label, labels, dir_labels):
    scaler, pca = scaler_pca
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca, file_label, labels, dir_labels

if type(args.scaler_pca) == str:
    scaler_pca = pickle.load(open(args.scaler_pca,'rb'))
else:
    scaler_pca = pickle.load(sys.stdin.buffer)


# Filter input files based on CSV if provided
if args.csv and args.sample_n is not None:
    try:
        df = pd.read_csv(args.csv)
        column_name = f'Sample_{args.sample_n}'
        if column_name not in df.columns:
            raise ValueError(f"Column {column_name} not found in CSV file")
            
        # Get valid basenames from the specified Sample_N column
        valid_files = set()
        for cell in df[column_name]:
            if pd.notna(cell):
                # Assuming cell contains file basename (without extension/path)
                valid_files.add(str(cell).strip().split('_')[0])
        
        # Filter input files to only those whose basenames are in valid_files
        filtered_input_files = []
        for file in args.input_files:
            basename = os.path.splitext(os.path.basename(file))[0].split('_')[0]
            if basename in valid_files:
                filtered_input_files.append(file)
        
        args.input_files = filtered_input_files
        args.input_files += ['ictv_sample_Order_7']
        print(f"Filtered to {len(args.input_files)} files based on {column_name}")
        print(f"Amount of valid files: {len(valid_files)}")

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)
   

activations = []
# Initialize empty lists for results
X_pca_results = []
file_labels_results = []
labels_results = []
dir_labels_result = []

for file in tqdm.tqdm(args.input_files):
    if file.split('.')[-1] == 'pkl':
        with open(file, 'rb') as f:
            loaded_dict = pickle.load(f)
    elif file.split('.')[-1] == 'json':
        with open(file, 'r') as f:
            loaded_dict = json.load(f)

    label = file.split('/')[-1].split('_layer')[0]
    dir_label = file.split('/')[-3]
    file_labels = [label for _ in loaded_dict.keys()]
    dir_labels = [dir_label for _ in loaded_dict.keys()]
    current_labels = list(loaded_dict.keys())
    activations = list(loaded_dict.values())
    
    # Process each file's data immediately
    try:
        X_pca, file_label, labels, dir_labels2 = pca_transform(scaler_pca, np.array(activations), file_labels, current_labels, dir_labels)
    except Exception as e:
        print(f'Problem with file {file}')
        continue
    
    # Append results
    X_pca_results.append(X_pca)
    file_labels_results.extend(file_label)
    labels_results.extend(labels)
    dir_labels_result.extend(dir_labels2)

# Combine all results
X_pca_final = np.concatenate(X_pca_results, axis=0)
output_data = [X_pca_final, file_labels_results, labels_results, dir_labels_result]

if type(args.output) == str:
    pickle.dump(output_data, open(args.output, "wb"))
else:
    pickle.dump(output_data, args.output)
