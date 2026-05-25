import argparse
import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
import pathlib
import glob # Needed for recursive file search

def create_subsequences(input_file, output_dir, window_size, step_size, max_seqs_per_file=1000):
    """
    Creates subsequences from a FASTA file and writes them in batches 
    to multiple FASTA files inside the specified output directory.
    
    This function remains mostly the same, but it's called with the specific 
    output directory for the *input_file*.
    """

    # 1. Determine the base name for output files
    # E.g., for input 'data/SRR123.fa', the basename is 'SRR123'
    input_basename = pathlib.Path(input_file).stem

    # Ensure the output directory exists
    # The calling function (main) is now responsible for ensuring the specific 
    # output_dir (e.g., output_root/dir1) exists. This check is kept as a safeguard.
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Subsequences from {pathlib.Path(input_file).name} will be saved in: {output_dir}")

    # Read the input FASTA file
    try:
        # Use SeqIO.parse for memory efficiency if files are large, and convert to list only if necessary later
        sequences = SeqIO.parse(input_file, "fasta")
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}", file=sys.stderr)
        return

    # Prepare to batch output sequences
    batch_records = []
    file_counter = 1

    def write_batch(records, base_name, counter, output_path):
        """Writes a batch of SeqRecords to a file."""
        # Construct the full output filename within the directory
        # Format: SRR123_sub_1.fa, SRR123_sub_2.fa, etc.
        output_filename = pathlib.Path(output_path) / f"{base_name}_sub_{counter}.fa"

        try:
            with open(output_filename, "w") as out_handle:
                SeqIO.write(records, out_handle, "fasta")
            print(f"Written {len(records)} sequences to {output_filename}")
        except IOError as e:
            print(f"Error writing to output file {output_filename}: {e}", file=sys.stderr)

    # Process each sequence
    for record in sequences:
        seq_len = len(record.seq)
        if seq_len < window_size:
            continue

        # Create subsequences using sliding window
        for start in range(0, seq_len - window_size + 1, step_size):
            end = start + window_size
            subseq = record.seq[start:end]

            # Skip sequences containing 'N'
            if 'N' in subseq.upper():
                continue

            # Construct new sequence ID
            # ID format: SRR123_originalID||start-end
            # Removed original ID from new_id to keep it cleaner, as the input_basename 
            # provides the primary source. New format: SRR123||start-end
            new_id = f"{input_basename}||{start+1}-{end}"
            new_record = SeqRecord(seq=subseq, id=new_id, description="")
            batch_records.append(new_record)

            # If batch is full, write to file and reset batch
            if len(batch_records) >= max_seqs_per_file:
                write_batch(batch_records, input_basename, file_counter, output_dir)
                batch_records = []
                file_counter += 1

    # Write remaining sequences in the last batch if any
    if batch_records:
        write_batch(batch_records, input_basename, file_counter, output_dir)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create subsequences from FASTA files in multiple directories using a sliding window and output to corresponding subdirectories in a root output path."
    )
    
    # Changed input to accept multiple directories
    parser.add_argument(
        "-i", 
        "--input_dirs", 
        nargs='+', # Accepts one or more arguments
        required=True, 
        help="One or more input directories containing FASTA files."
    )
    
    # New argument for file suffix
    parser.add_argument(
        "-x", 
        "--suffix", 
        required=True, 
        help="Suffix of the FASTA files (e.g., .fa or .fasta)."
    )
    
    # Changed output to be the root output directory
    parser.add_argument(
        "-o",
        "--output_root",
        required=True,
        help="Root output directory path. Subdirectories corresponding to input_dirs will be created here."
    )
    
    parser.add_argument("-w", "--window", type=int, required=True, help="Window size (subsequence length).")
    parser.add_argument("-s", "--step", type=int, required=True, help="Step size for sliding window.")

    # Optional: allow changing the maximum sequences per file
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=1000,
        help="Maximum number of subsequences per output FASTA file."
    )

    args = parser.parse_args()

    # --- Main Logic for Multiple Directories ---
    
    # Ensure the root output directory exists
    pathlib.Path(args.output_root).mkdir(parents=True, exist_ok=True)
    
    # Process each input directory
    for input_dir_str in args.input_dirs:
        input_dir = pathlib.Path(input_dir_str)
        
        # 1. Determine the corresponding output directory
        # The basename of the input directory (e.g., 'dir1' from 'data/dir1')
        input_dir_basename = input_dir.name 
        
        # The specific output directory for this batch of files (e.g., 'output_root/dir1')
        output_dir = pathlib.Path(args.output_root) / input_dir_basename
        
        # Ensure the output subdirectory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing files in: {input_dir}")
        print(f"Outputting to: {output_dir}")
        
        # 2. Find FASTA files with the specified suffix in the current input directory
        # Use glob to find all files recursively matching the suffix
        search_pattern = str(input_dir / f"**/*{args.suffix}")
        fasta_files = glob.glob(search_pattern, recursive=True)

        if not fasta_files:
            print(f"Warning: No files found with suffix '{args.suffix}' in '{input_dir_str}'.")
            continue

        # 3. Process each FASTA file found
        for fasta_file in fasta_files:
            # Call the function with the specific file and the specific output subdirectory
            create_subsequences(
                input_file=fasta_file, 
                output_dir=output_dir, 
                window_size=args.window, 
                step_size=args.step, 
                max_seqs_per_file=args.max_seqs
            )

if __name__ == "__main__":
    main()
