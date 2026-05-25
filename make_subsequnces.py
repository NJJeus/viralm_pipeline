import argparse
import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
import pathlib

def create_subsequences(input_file, output_dir, window_size, step_size, max_seqs_per_file=1000):
    """
    Creates subsequences from a FASTA file and writes them in batches 
    to multiple FASTA files inside the specified output directory.
    """
    
    # 1. Determine the base name for output files
    # E.g., for input 'data/SRR123.fa', the basename is 'SRR123'
    input_basename = pathlib.Path(input_file).stem
    
    # Ensure the output directory exists
    if output_dir != sys.stdout:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Subsequences will be saved in: {output_dir}")

    # Read the input FASTA file
    try:
        sequences = list(SeqIO.parse(input_file, "fasta"))
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}", file=sys.stderr)
        return

    # Prepare to batch output sequences
    batch_records = []
    file_counter = 1

    def write_batch(records, base_name, counter, output_path):
        """Writes a batch of SeqRecords to a file."""
        if output_path == sys.stdout:
            # Writing to stdout (only possible if a single file is expected)
            SeqIO.write(records, sys.stdout, "fasta")
        else:
            # Construct the full output filename within the directory
            # Format: SRR123_sub_1.fa, SRR123_sub_2.fa, etc.
            # Using a fixed suffix like '_sub' for clarity
            output_filename = pathlib.Path(output_path) / f"{base_name}_sub_{counter}.fa"
            
            with open(output_filename, "w") as out_handle:
                SeqIO.write(records, out_handle, "fasta")
            print(f"Written {len(records)} sequences to {output_filename}")

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
            new_id = f"{input_basename}_{record.id}||{start+1}-{end}"
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
        description="Create subsequences from a FASTA file using a sliding window approach and output to a specified directory."
    )
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file path.")
    parser.add_argument(
        "-o", 
        "--output", 
        required=True, # Made required to enforce directory output in Snakemake context
        help="Output directory path where subsequence FASTA files will be saved (e.g., intermediate/SRR123_subsequences)."
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

    # Call the function with parsed arguments
    create_subsequences(args.input, args.output, args.window, args.step, args.max_seqs)

if __name__ == "__main__":
    main()
