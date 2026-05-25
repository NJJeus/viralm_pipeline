import argparse
import glob
import pickle
import numpy as np
import pandas as pd
import os

def process_data(input_dir, reference, output_path, layers):
    """
    Loads, processes, and saves data from pickled files based on a list of layers and reference.

    Args:
        input_dir (str): The directory containing the pickle files.
        reference (str): The base name of the reference file (e.g., 'A').
        output_path (str): The path to save the final CSV file.
        layers (list): A list of integer layer numbers to process.
    """
    data_params = []
    data_results = []

    # Iterate over the list of layers read from arguments
    for layer in layers: 
        # Construct the glob pattern
        search_pattern = os.path.join(input_dir, f'layer_{layer}_ref_{reference}.pkl')
        
        for file_path in glob.glob(search_pattern):
            print(f"Loading file: {file_path}")
            try:
                # Use 'rb' for reading binary files (pickle)
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
                
                print(np.array(file_data[1]).shape)
                # Prepend the 'Layer' column
                layer_column = np.full((len(file_data[0]), 1), layer)
                data_params.append(np.column_stack([layer_column, file_data[0]]))

                
                # Append results array
                data_results.append(file_data[1])

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    if not data_params or not data_results:
        print("No data files were loaded. Exiting.")
        return

    # Concatenate all collected data
    data_params = np.concatenate(data_params)
    data_results = np.concatenate(data_results)

    # --- Data Processing and Filtering ---

    data_array = np.concatenate([data_params, data_results], axis=1)
    
    columns = [
        'Layer', 'n_chunks', 'n_components', 'v_th', 'Sensitivity', 
        'Precision', 'n1', 'n2', 'n3', 'n4' , 'tp', 'fp', 'tn', 'fn'
    ]
    data = pd.DataFrame(data_array, columns=columns)
    
    data_all = data.copy()

    # Filter for 'v_th' == 0
    data_all = data_all[(data_all['v_th'] == 0.0)] 

    # Calculate F1 score
    data_all['f1_score'] = (2 * data_all.Sensitivity * data_all.Precision) / (data_all.Precision + data_all.Sensitivity)

    # Convert 'Layer' to integer then to string
    data_all['Layer'] = data_all['Layer'].astype(int).astype(str, copy=True)
    
    # Sort the data by f1_score
    data_all = data_all.sort_values('f1_score', ascending=False)

    # Save the final DataFrame
    data_all.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to: {output_path}")


if __name__ == '__main__':
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Process layer-based pickled data and calculate F1 scores.")
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help='The input directory containing the pickle files.'
    )
    parser.add_argument(
        '--reference', 
        type=str, 
        required=True, 
        help='The reference basename used in the filenames (e.g., A).'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='The full path where the final CSV output file will be saved.'
    )
    parser.add_argument(
        '--layers', 
        type=int, 
        nargs='+', # This tells argparse to expect one or more values
        required=True, 
        help='A space-separated list of layer numbers to process (e.g., 2 4 6 8 10 12).'
    )

    args = parser.parse_args()

    # Pass the list of layers to the main function
    process_data(args.input_dir, args.reference, args.output_path, args.layers)
