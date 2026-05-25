##
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import glob
import os
import pandas as pd
import seaborn as sns
sns.set_theme()
import random
import pickle

import tqdm

from scipy.stats import fisher_exact

import numpy as np
from sklearn import metrics
from itertools import product
from collections import defaultdict

import argparse

def calculate_adaptive_hypercube_density(points, num_hypercubes_per_dim, adaptive=True, return_dots=False , mins=None, maxs=None):
    """
    Adaptive version with percentile-based binning.
    
    Args:
        adaptive: When True, uses percentile-based binning
    """
    # Convert to list if single number
    if isinstance(num_hypercubes_per_dim, int):
        num_hypercubes = [num_hypercubes_per_dim] * points.shape[1]
    else:
        num_hypercubes = num_hypercubes_per_dim

    bin_edges = []
    hypercube_indices = np.zeros_like(points, dtype=int)

    for dim in range(points.shape[1]):
        if adaptive:
            # Calculate percentile-based edges
            percentiles = np.linspace(0, 100, num_hypercubes[dim] + 1)
            edges = np.percentile(points[:, dim], percentiles)
            
            # Prevent edge cases by slightly expanding range
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            
            bin_edges.append(edges)
            
            # Digitize with clipping
            dim_indices = np.digitize(points[:, dim], edges) - 1
            hypercube_indices[:, dim] = np.clip(dim_indices, 0, num_hypercubes[dim]-1)
        else:
            # Original fixed grid logic
            mn = np.min(points[:, dim])
            mx = np.max(points[:, dim])
            scaled = (points[:, dim] - mn) / (mx - mn) * num_hypercubes[dim]
            hypercube_indices[:, dim] = np.clip(scaled.astype(int), 0, num_hypercubes[dim]-1)

    # Count occurrences (same as original)
    unique_coords, counts = np.unique(hypercube_indices, axis=0, return_counts=True)
    output_shape = tuple(num_hypercubes)
    output = np.zeros(output_shape, dtype=int)
    
    for coord, count in zip(unique_coords, counts):
        output[tuple(coord)] = count

    return (output, hypercube_indices) if return_dots else output


def calculate_hypercube_dot_density(points, num_hypercubes_per_dim, mins=None, maxs=None, return_dots=False):
    """
    Calculate the number of dots in each hypercube when dividing the N-dimensional space.
    
    Args:
        points: numpy array of shape (X, N) where X is the number of points (dots)
                and N is the dimensionality (2 in your case)
        num_hypercubes_per_dim: number of hypercubes to create along each dimension
                               (can be a single integer or a list with one value per dimension)
    
    Returns:
        A numpy array with the count of dots in each hypercube
    """
    # Convert single number to list if needed
    if isinstance(num_hypercubes_per_dim, int):
        num_hypercubes_per_dim = [num_hypercubes_per_dim] * points.shape[1]
    
    # Find min and max values in each dimension to define the space
    if mins is None:
        mins = np.min(points, axis=0)
    if maxs is None:
        maxs = np.max(points, axis=0)
    
    # Calculate which hypercube each point falls into
    hypercube_indices = np.zeros_like(points, dtype=int)
    for dim in range(points.shape[1]):
        # Scale points to [0, num_hypercubes_per_dim) range
        scaled = (points[:, dim] - mins[dim]) / (maxs[dim] - mins[dim]) * num_hypercubes_per_dim[dim]
        # Convert to integer indices (clipping to handle edge cases)
        hypercube_indices[:, dim] = np.clip(scaled.astype(int), 0, num_hypercubes_per_dim[dim] - 1)
    
    # Count occurrences of each hypercube index combination
    unique_coords, counts = np.unique(hypercube_indices, axis=0, return_counts=True)
    
    # Create the output array filled with zeros
    output_shape = tuple(num_hypercubes_per_dim)
    output = np.zeros(output_shape, dtype=int)
    
    # Fill in the counts
    for coord, count in zip(unique_coords, counts):
        output[tuple(coord)] = count

    if return_dots:
        return output, hypercube_indices
    else:
        return output

def plot_comparison(df1, df2, vmax=None, d1_name=None, d2_name=None, d3=None, subtitle=''):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    df1, df2 = collapse_N_D_array(df1),  collapse_N_D_array(df2)
    fig.suptitle(subtitle)
    sns.heatmap(df1, ax=axs[0], vmax=vmax)
    axs[0].set_title(d1_name)
    sns.heatmap(df2, ax=axs[1], vmax=vmax)
    axs[1].set_title(d2_name)

    if not d3 is None:
        d3 = collapse_N_D_array(d3)
        sns.heatmap(d3, cmap="crest", ax=axs[2])
        axs[2].set_title("Two-tailed KS test p < 0.01/ncells")
    else:
        
        sns.heatmap(df2/df2.max().max()-df1/df1.max().max(), cmap=sns.color_palette("vlag", as_cmap=True), ax=axs[2])
        axs[2].set_title(f"{d1_name} vs {d2_name}")



    
    
def plot_pca(X_pca_labels, layer, class_mapping=None, point_size=30, alpha=0.7, figsize=(10, 8)):
     # Create plot
    
    X_pca, labels = X_pca_labels
    
    class_labels = [] if class_mapping else None
    if class_mapping:
        for seq_name in labels:
            class_labels.append(class_mapping.get(seq_name, "Unknown"))
    plt.figure(figsize=figsize)

    if class_mapping:
        # Class-based coloring
        unique_classes = sorted(set(class_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        
        for cls, color in zip(unique_classes, colors):
            mask = np.array(class_labels) == cls
            plt.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1],
                color=color,
                label=cls,
                s=point_size,
                alpha=alpha
            )
        plt.legend(title="Classes")
    else:
        # Sequence-based coloring
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            plt.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1],
                color=color,
                label=label,
                s=point_size,
                alpha=alpha
            )
        plt.legend(title="Sequences", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add variance explained
    plt.title(f"PCA of Layer {layer}")
    plt.xlabel(f"Principal Component 1")
    plt.ylabel(f"Principal Component 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def ks_space(d1, d2, sample_size=500, n_samples=50, ps=20, alpha=None):
    # Get the number of dimensions from the input data
    n_dims = d1.shape[1]
    
    # Generate random samples for both datasets
    d1s = [d1[np.random.randint(d1.shape[0], size=sample_size), :] for _ in range(n_samples)]
    d2s = [d2[np.random.randint(d2.shape[0], size=sample_size), :] for _ in range(n_samples)]
    
    # Calculate hypercube dot densities for all samples
    mins = np.min(np.vstack((d1, d2)), axis=0)
    maxs = np.max(np.vstack((d1, d2)), axis=0)
    
    d1s = [calculate_hypercube_dot_density(d, ps, mins=mins, maxs=maxs) for d in d1s]
    d2s = [calculate_hypercube_dot_density(d, ps, mins=mins, maxs=maxs) for d in d2s]

    # Initialize result arrays with shape based on number of dimensions
    result_shape = [ps] * n_dims
    ress_v = np.zeros(result_shape)
    ress_p = np.zeros(result_shape)
    
    # Create multi-index iterator for all dimensions

    indices = product(*[range(n_dims) for _ in range(n_dims)])
    
    # Perform KS test for each hypercube
    for idx in indices:
        x = [d1s[s][idx] for s in range(n_samples)]
        y = [d2s[s][idx] for s in range(n_samples)]
        ress_v[idx] = kstest(x, y)[0]
        ress_p[idx] = kstest(x, y)[1]

    # Apply multiple testing correction if alpha is not provided
    if alpha is None:
        alpha = (0.05 / (ps**n_dims))
    
    # Mask non-significant results
    ress_v[ress_p > alpha] = 0

    return ress_v, ress_p


def fisher_space(d1_hyper, d2_hyper, alpha=None):
    n_dims = d1_hyper.ndim
    ps = d1_hyper.shape[0]

    # Initialize result arrays with shape based on number of dimensions
    result_shape = d1_hyper.shape
    ress_v = np.zeros(result_shape)
    ress_p = np.zeros(result_shape)
    indices = product(*[range(ps) for i in range(n_dims)])
    # Perform KS test for each hypercube
    for idx in indices:
        d1_in = d1_hyper[idx]
        d1_out = d1_hyper.sum() - d1_in
        d2_in = d2_hyper[idx]
        d2_out = d2_hyper.sum() - d2_in

        ftest = fisher_exact([[d2_in, d2_out], [d1_in, d1_out]], alternative='greater')
        ress_v[idx] = ftest[0]
        ress_p[idx] = ftest[1]
    # Apply multiple testing correction if alpha is not provided
    if alpha is None:
        alpha = (0.05 / (ps**n_dims))
    
    # Mask non-significant results
    ress_v[ress_p > alpha] = 0

    return ress_v, ress_p




def collapse_N_D_array(arr):
    return arr.mean(axis=tuple(range(2, arr.ndim))) if arr.ndim > 2 else arr

##
parser = argparse.ArgumentParser(description='Proceasss some arguments.')
    
# Add arguments
parser.add_argument('--input_file', type=str, required=True,
                   help='Path to the input file')
parser.add_argument('--output_file', type=str, required=True,
                   help='Path to the output file')
parser.add_argument('--reference_name', type=str)
parser.add_argument('--blast_dir', type=str)
parser.add_argument('--labels', nargs='*', help='LABELS')

# Parse the arguments
args = parser.parse_args()
##
"""
class Args:
    def __init__(self):
        self.input_file = '04_pca_data/pca_data_6.pkl'
        self.output_file = 'test.csv'
        self.reference_name = 'someSars'


args = Args()
"""
##

senses = []
specifs = []

iter_through = list(product([10, 15, 20, 25], [2, 3, 4, 5], [0]))
res_list = []
layer_prev = 0
n_comp_prev = 0
sample_prev = -1

X_pca_labels_all = pickle.load(open(args.input_file, 'rb'))
    
X_pca_labels_all[1] = X_pca_labels_all[3]

entry_counts = defaultdict(int)
new_labels = []

X_pca_labels = X_pca_labels_all

for entry in X_pca_labels[2]:
    entry_counts[entry] += 1
    count = entry_counts[entry]
    
    prefix, suffix = entry.split('||', 1) if len(entry.split('||', 1)) == 2 else (entry, '')
    new_prefix = f"{prefix}_x{count}||"
    new_entry = new_prefix + suffix
    new_labels.append(new_entry)    
    
X_pca_labels[2] = new_labels
        
labs = list(args.labels)
print("LABS:", labs)
print(labs)
blastn_columns = ['qseqid',
 'sseqid',
 'pident',
 'length',
 'mismatch',
 'gapopen',
 'qstart',
 'qend',
 'sstart',
 'send',
 'evalue',
 'bitscore',
 'qlen']

corona_contigs = []
reference_name = args.reference_name
print('Referenec_name', args.blast_dir)
for file in glob.glob(f'{args.blast_dir}/*_{reference_name}.tsv'):
    print(file)
    try:
        blastn_data = pd.read_table(file, header=None)
        blastn_data.columns = blastn_columns
    except Exception as e:
        print(e)
        continue
    blastn_data['salignper'] = blastn_data.length / blastn_data.qlen
    contigs = blastn_data.query('salignper > 0.8 & qlen >= 1000 & pident > 95').drop_duplicates('qseqid').qseqid.to_list()
    print('Contigs:', contigs)
    file_name = file.split('/')[-1].split('_vs_')[0]
    corona_contigs += [file_name+'_'+i for i in contigs]

##

for ps, n_components, v_th in tqdm.tqdm(iter_through):
    if n_components != n_comp_prev or ps != ps_prev:
        X_pca_labels[0] = X_pca_labels_all[0][:, :n_components]
        
        print('p1', X_pca_labels[0].shape)
        n_comp_prev = n_components
        ps_prev = ps
    
        mins = np.min(X_pca_labels[0], axis=0)
        maxs = np.max(X_pca_labels[0], axis=0)
    
        d1_idx = (np.array(X_pca_labels[1]) == labs[0])
        d2_idx = (np.array(X_pca_labels[1]) == labs[2])
        d1 = X_pca_labels[0][d1_idx, :]
        d2 = X_pca_labels[0][d2_idx, :]

        print('p2', d2.shape)

        d1_seq_labels = np.array(X_pca_labels[2])[d1_idx]
        d2_seq_labels = np.array(X_pca_labels[2])[d2_idx]
    
    
        d1_hyper, d1_dots = calculate_hypercube_dot_density(d1, ps, mins=mins, maxs=maxs, return_dots=True)
        d2_hyper, d2_dots = calculate_hypercube_dot_density(d2, ps, mins=mins, maxs=maxs, return_dots=True)

        print('p3', d2_dots.shape)

        where_viruses = ((d2_hyper/d2_hyper.max().max() - d1_hyper/d1_hyper.max().max()) > 0)
    
    
        d2_idx = (np.array(X_pca_labels[1]) == labs[1])
        d2 = X_pca_labels[0][d2_idx, :]
        d2_seq_labels_all = np.array(X_pca_labels[2])[d2_idx]
        d2_hyper_all, d2_dots_all = calculate_hypercube_dot_density(d2, ps, mins=mins, maxs=maxs, return_dots=True)
    
    
        d1_idx = (np.array(X_pca_labels[1]) == labs[0])
        d2_idx = (np.array(X_pca_labels[1]) == labs[1])

        d1 = X_pca_labels[0][d1_idx, :]
        d2 = X_pca_labels[0][d2_idx, :]
        test_var = set(np.array(X_pca_labels[1]))

        print('p4', labs[1], set(X_pca_labels[1]), test_var, args.labels)

        d1_seq_labels = np.array(X_pca_labels[2])[d1_idx]
        d2_seq_labels = np.array(X_pca_labels[2])[d2_idx]
    
        d1_hyper, d1_dots = calculate_hypercube_dot_density(d1, ps, mins=mins, maxs=maxs, return_dots=True)
        d2_hyper, d2_dots = calculate_hypercube_dot_density(d2, ps, mins=mins, maxs=maxs, return_dots=True)
    
        print('p5', d2_dots.shape)
        where_more = ((d2_hyper/d2_hyper.max().max() - d1_hyper/d1_hyper.max().max()) > 0)


        vmax = max(d1_hyper.max().max(), d2_hyper.max().max())
        print('p6', vmax)
        ress_v, ress_p = fisher_space(d1_hyper, d2_hyper, alpha=0.05/ps**d1.shape[1])

        print('p7', ress_p[:5])
    try:
        s = []
        for coord in np.argwhere((ress_v > v_th) * where_more * where_viruses):
            s.append(d2_seq_labels_all[(d2_dots_all[:, :] == coord).all(1)])
        output_labels = np.concatenate(s)
        pd.Series(list(output_labels)).to_csv(f"{args.output_file}_{ps}_{n_components}_{v_th}.csv", index=False)
        output_labels = set([i.split('|')[0] for i in output_labels])
    except Exception:
        res_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        continue
        
    
    corona_in_dataset = set([i.split('|')[0] for i in X_pca_labels[2] if i.split('_x1')[0] in corona_contigs])
    print('p7', corona_contigs[:3], X_pca_labels[2][:3])
    n_all_labels = len(set([i.split('|')[0] for i in X_pca_labels[2]]))
    n_positive = len(output_labels)
    n_corona_count =  len(corona_in_dataset)
    n_corona_in_out = len(set([i for i in output_labels if i in corona_in_dataset]))

    tp = n_corona_in_out
    fn = len(corona_in_dataset) - tp
    tn = n_all_labels - n_positive - fn
    fp = n_positive - tp 
    if n_corona_count == 0:
        n_corona_count = 100000
    sensitivity = round(n_corona_in_out / n_corona_count*100, 2)
    precision = round(n_corona_in_out / n_positive*100, 2)
    res_list.append([sensitivity, precision, n_all_labels, n_positive, n_corona_count, n_corona_in_out, tp, fp, tn, fn])


##    


with open(f'{args.output_file}', 'wb') as handle:
    pickle.dump((iter_through, res_list), handle)

