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
import json
import tqdm


##
parser = argparse.ArgumentParser(description='Process input files and write to output file or stdout.')

# Unlimited input files (nargs='*' for zero or more, '+' for one or more)
parser.add_argument('input_files', nargs='*', help='Input file(s)')
parser.add_argument('-n', '--n_components', type=int, default=2)

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

##

def preprocee_hidden_layers(hidden_states_dict, layer):
    activations = []
    labels = []
    for seq_name, segments in hidden_states_dict.items():
        for segment in segments:
            # Get activations for specified layer
            if layer < segment.shape[0]:
                layer_activations = segment[layer]
            else:
                print(f'The layer is selected as {layer} from {segment.shape[0]} in {seq_name}')
                continue
            activations.append(layer_activations)
            labels.append(seq_name)
    return activations, labels


def fit_pca(activations_labels,
                  n_components=2):

    activations, labels = activations_labels

    X = np.array(activations)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca = pca.fit(X)


    return scaler, pca  # Return PCA object for further analysis 


##

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
        print(f"Filtered to {len(args.input_files)} files based on {column_name}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)
activations = []

for file in tqdm.tqdm(args.input_files):
    if file.split('.')[-1] == 'pkl':
        with open(file, 'rb') as f:
            loaded_dict = pickle.load(f)
            activations = activations + list(loaded_dict.values())
    elif file.split('.')[-1] == 'json':
        with open(file, 'r') as f:
            try:
                loaded_dict = json.load(f)
                activations = activations + list(loaded_dict.values())
            except Exception:
                print(f'Error on {file}')


activations_labels = activations, []
scaler_pca = fit_pca(activations_labels, n_components=args.n_components)

print(scaler_pca[1].explained_variance_ratio_)

##
if type(args.output) == str:
    pickle.dump(scaler_pca, open(args.output, "wb"))
else:
    pickle.dump(scaler_pca, args.output)
