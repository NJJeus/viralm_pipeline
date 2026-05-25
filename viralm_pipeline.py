# Snakefile (Revised Target Generation)

import os
from snakemake.io import glob_wildcards, expand

# --- Configuration Loading ---
INPUT_DIRS = config["input_dir"]
LAYERS = config["layers_to_monitor"]
OUTPUT_BASE_DIR = "results"
VIRALM_PKL_SUBDIR = "viralm_pkl"
input_base = config['input_base']

# --- Wildcard Definition ---

# 1. Get the base names for the input directories (equivalent to DIR_IDS)
# e.g., ['input_set1', 'input_set2']
INPUT_SET_NAMES = [os.path.basename(d) for d in INPUT_DIRS]

# 2. Extract sample names and their source directory identifier ({set_name})
SAMPLES_PER_SET = {}
# Use glob_wildcards to find all samples in all input directories
for full_path in INPUT_DIRS:
    set_name = os.path.basename(full_path)
    # The pattern is the full path prefix + wildcard {sample} + suffix (.fa)
    samples_in_dir, = glob_wildcards(full_path + os.sep + "{sample}.fa")
    
    # Store samples associated with their set name
    SAMPLES_PER_SET[set_name] = samples_in_dir

# 3. Create a combined list of all unique sample names across all sets
ALL_SAMPLES = []
for samples in SAMPLES_PER_SET.values():
    ALL_SAMPLES.extend(samples)
# Filter for unique samples if needed, but for target generation, we use the set-specific lists.

print("Input Set Names:", INPUT_SET_NAMES)
print("Samples per Set:", SAMPLES_PER_SET)
print("Layers to monitor:", LAYERS)

# --- Target Definition (All Rule) using expand ---

# Output pattern structure: 
# {OUTPUT_BASE_DIR}/{set_name}/{VIRALM_PKL_SUBDIR}/{sample}/{sample}_layer_{layer}.pkl
TARGET_PATH_PATTERN = os.path.join(
    OUTPUT_BASE_DIR, 
    "{set_name}", 
    VIRALM_PKL_SUBDIR, 
    "{sample}", 
    "{sample}_layer_{layer}.pkl"
)

# Generate a list of all required output files using expand
target_outputs = []
for set_name in INPUT_SET_NAMES:
    # Use expand for the Cartesian product of SAMPLES and LAYERS for the current set
    expanded_targets = expand(
        TARGET_PATH_PATTERN,
        set_name=set_name,             # The directory name, e.g., 'input_set1'
        sample=SAMPLES_PER_SET[set_name], # The samples found in that directory
        layer=LAYERS                   # The layers from config
    )
    target_outputs.extend(expanded_targets)

# The 'all' rule uses all generated file paths as its input
rule all:
    input:
        target_outputs

# --- Rule for Running the Script ---
# Inside Snakefile

# Inside Snakefile (Revised Rule using params)

rule viralm_processing:
    input:
        lambda wildcards: os.path.join(
                input_base, 
                wildcards.set_name, 
                f"{wildcards.sample}.fa"
                )
    output:
        pkl = os.path.join(
                OUTPUT_BASE_DIR, 
                "{set_name}", 
                VIRALM_PKL_SUBDIR, 
                "{sample}", 
                "{sample}_layer_{layer}.pkl"
                )
        
    params:
        output_dir_for_script = os.path.join(
                OUTPUT_BASE_DIR, 
                "{set_name}", 
                VIRALM_PKL_SUBDIR, 
                "{sample}"
                )
    conda:
        'viralm.yaml'

    shell:
        """
        # Create the output directory based on the 'params' value
        mkdir -p {params.output_dir_for_script} && \\
        
        # Pass the directory path using the 'params' variable
        python viralm_hook.py \\
        --input_pth {input} \\
        --output_pth {params.output_dir_for_script} \\
        --batch_size 1 \\
        --layers_to_monitor {wildcards.layer}
        
        # Note: The script viralm_hook.py MUST now create the file 
        # '{params.output_dir_for_script}/{wildcards.sample}_layer_{wildcards.layer}.pkl'
        """
