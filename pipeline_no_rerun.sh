#!/bin/bash
#
# Automated Bash Pipeline for Sequence Processing, ViRALM Analysis, and PCA.
#
# NOTE: This script assumes that 'make_subsequnces.py', 'viralm_hook.py', and
# 'pca_hidden_states.py' are accessible in your PATH, and that the 'viralm2'
# conda environment has been created and contains the necessary dependencies
# for the ViRALM hook script.

# --- Configuration Variables ---

# Define the base input path and the array of source directories.
BASE_INPUT_PATH="./TEST_DATA/"
SOURCES=("DSI" "DSII" "DSII") # All input sources for Step 1 & 2
PCA_SOURCES=("${SOURCES[@]:0:2}")             # Sources to be included in Step 3 (PCA)

TARGET_DIR="TEST_DATA/DSII/"

OUTPUT_BASE="TEST_OUTPUT/"

# Define your output directories here.
SPLIT_DIR="${OUTPUT_BASE}/01_split_sequences"
VIRALM_OUT_DIR="${OUTPUT_BASE}/02_viralm_outputs"
PCA_MODELS_DIR="${OUTPUT_BASE}/03_pca_models"
PCA_DATA_DIR="${OUTPUT_BASE}/04_pca_data"
BLASTN_DIR="${OUTPUT_BASE}/05_blastn_to_reference"
PKL_POSITIVE_CONTIGS="${OUTPUT_BASE}/06_pkl_positive_contigs"
OUTPUT_REPORT_DIR="${OUTPUT_BASE}/07_reports" # Defined here for use in Step 7
CONDA_ENV="viralm2"
declare -a LAYERS_TO_MONITOR=(2 4 6 8 10 12)

# --- Setup ---

echo "--- 1. Setting up directories ---"
# Create base output directories
mkdir -p "${SPLIT_DIR}"
mkdir -p "${VIRALM_OUT_DIR}"
mkdir -p "${PCA_MODELS_DIR}"
mkdir -p "${PCA_DATA_DIR}"
mkdir -p "${BLASTN_DIR}"
mkdir -p "${PKL_POSITIVE_CONTIGS}"
mkdir -p "$OUTPUT_REPORT_DIR"

# Create source-specific subdirectories
for source in "${SOURCES[@]}"; do
    mkdir -p "${SPLIT_DIR}/${source}"
    mkdir -p "${VIRALM_OUT_DIR}/${source}"
    echo "Created directories: ${SPLIT_DIR}/${source} and ${VIRALM_OUT_DIR}/${source}"
done
echo "Directories created or verified."

# --- Step 1: Split Long Sequences into Subsequences (Per File Idempotency) ---
# Runs make_subsequnces.py for all .fa files in each source directory.

echo "--- 2. Running make_subsequnces.py (Step 1 of 3) ---"
# Loop over all defined sources
for source in "${SOURCES[@]}"; do
    INPUT_DIR="${BASE_INPUT_PATH}/${source}"
    echo "--- Processing source directory: ${INPUT_DIR} ---"

    # Inner loop over every .fa file in the current input directory
    for input_file in "${INPUT_DIR}"/*.fa; do
        if [ -f "$input_file" ]; then
            filename=$(basename "$input_file")
            output_path="${SPLIT_DIR}/${source}/${filename}"
            
            # IDEMPOTENCY CHECK (Per File): Check if the output directory for this split file exists
            if [ -d "$output_path" ]; then
                echo "Skipping Step 1 for $filename (Source: $source). Output directory already exists: $output_path"
                continue # Skip to the next file
            fi

            echo "Processing: $filename (Source: $source)"
            python make_subsequnces.py \
                -i "$input_file" \
                -w 1000 \
                -s 1000 \
                -o "$output_path"
        fi
    done
done
echo "make_subsequnces.py completed for all sources."

# --- Step 2: Run ViRALM Hook on Split Sequences (Per Source Directory Idempotency) ---
# Runs viralm_hook.py using the specified conda environment.

echo "--- 3. Running viralm_hook.py inside the ${CONDA_ENV} environment (Step 2 of 3) ---"
# Loop over all defined sources
for source in "${SOURCES[@]}"; do
    SPLIT_SOURCE_DIR="${SPLIT_DIR}/${source}"
    VIRALM_SOURCE_DIR="${VIRALM_OUT_DIR}/${source}"
    
    # IDEMPOTENCY CHECK (Per Source Directory): Check if the source output directory is non-empty
    if [ -n "$(find "$VIRALM_SOURCE_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        echo "Skipping ViRALM hook for Source: $source. Output directory $VIRALM_SOURCE_DIR is non-empty."
        continue
    fi

    echo "--- Running ViRALM on split files in: ${SPLIT_SOURCE_DIR} ---"

    # Loop over all files created in the source-specific split directory
    for split_file in "${SPLIT_SOURCE_DIR}"/*/*.fa; do
        if [ -f "$split_file" ]; then
            filename=$(basename "$split_file")
            output_dir_name="${filename%.fa}"
            output_path="${VIRALM_SOURCE_DIR}/${output_dir_name}"

            echo "Running ViRALM on: $filename (Source: $source), saving to ${output_path}/"

            conda run -n "${CONDA_ENV}" python viralm_hook.py \
                --input_pth "$split_file" \
                --output_pth "$output_path" \
                --batch_size 1 \
                --layers_to_monitor 1 2 3 4 5 6 7 8 9 10 11 12
        fi
    done
done
echo "viralm_hook.py completed for all sources."

# --- Step 3: PCA Analysis of Hidden States (File-level Idempotency) ---

echo "--- 4. Running pca_hidden_states.py (Step 3 of 3) on DSI and DSII data only ---"

# Train Model
echo "Train model \n"
for layer in "${LAYERS_TO_MONITOR[@]}"; do
	output_path="${PCA_MODELS_DIR}/pca_model_${layer}.pkl"

	# IDEMPOTENCY CHECK (Model File): Check if the PCA model file exists
	if [ -f "$output_path" ]; then
		echo "Skipping PCA Model Training for layer ${layer}. Model file already exists: $output_path"
		continue
	fi

	PCA_INPUT_GLOB=""
	    for source in "${PCA_SOURCES[@]}"; do
		PCA_INPUT_GLOB+="${VIRALM_OUT_DIR}/${source}/*/*_${layer}.pkl"
		PCA_INPUT_GLOB+=" "
	done
	echo -e " \n"
	echo $PCA_INPUT_GLOB
	python pca_hidden_states_sample.py $PCA_INPUT_GLOB -n 10 -o "$output_path"
done

# Transform Data
echo -e "Transform Data \n"
for layer in "${LAYERS_TO_MONITOR[@]}"; do
	output_path="${PCA_DATA_DIR}/pca_data_${layer}.pkl"

	# IDEMPOTENCY CHECK (Transformed Data File): Check if the transformed data file exists
	if [ -f "$output_path" ]; then
		echo "Skipping PCA Data Transformation for layer ${layer}. Output file already exists: $output_path"
		continue
	fi

	PCA_INPUT_GLOB=""
        for source in "${SOURCES[@]}"; do
            PCA_INPUT_GLOB+="${VIRALM_OUT_DIR}/${source}/*/*_${layer}.pkl"
            PCA_INPUT_GLOB+=" "
	done
	pca_model="${PCA_MODELS_DIR}/pca_model_${layer}.pkl"
	echo $pca_model
	echo -e "\n"
	echo "python pca_transform.py ${PCA_INPUT_GLOB}  -p ${pca_model} -o ${output_path}"

	python pca_transform.py $PCA_INPUT_GLOB  -p $pca_model -o $output_path
done

echo "--- 5. BLASTN ORIGINAL SEQUNCES (Per Source Directory Idempotency) -------"

for query_source in "${PCA_SOURCES[@]}"; do
        QUERY_DIR="${BASE_INPUT_PATH}/${query_source}"
        OUTPUT_SUBDIR="${BLASTN_DIR}/${query_source}" # Output path: ./05_blastn_to_reference/DSI/
        
        mkdir -p "${OUTPUT_SUBDIR}" # Create source-specific output directory

        # IDEMPOTENCY CHECK (Per Source Directory): Check if the source output directory is non-empty
        if [ -n "$(find "$OUTPUT_SUBDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
            echo "Skipping BLASTN for Source: $query_source. Output directory $OUTPUT_SUBDIR is non-empty."
            continue
        fi

	echo $BASE_INPUT_PATH
        echo "Processing Queries from: ${QUERY_DIR}"
        
        # 2. Inner loop through all .fa files in the Query directory
        for input_fasta in "${QUERY_DIR}"/*.fa; do
            if [ -f "$input_fasta" ]; then
                # Get the basename of the query file (e.g., sample_A.fa -> sample_A)
                input_basename=$(basename "$input_fasta" .fa)
                echo -e "Reference1 ${TARGET_DIR} \n"
		echo -e "${input_fasta}"
                # 3. Innermost loop through all .fa and .fna files in the Reference directory (Subjects)
                for reference_fasta in "${TARGET_DIR}"/*.fa "${TARGET_DIR}"/*.fna; do
                    if [ -f "$reference_fasta" ]; then
                        # Get the basename of the reference file (e.g., ref_genome_X.fna -> ref_genome_X)
                        reference_basename=$(basename "$reference_fasta")
                        reference_basename="${reference_basename%.*}" # Remove extension
                        
                        OUTPUT_FILE="${OUTPUT_SUBDIR}/${input_basename}_vs_${reference_basename}.tsv"
                        
                        echo "BLASTN: Query: ${input_basename} | Subject: ${reference_basename}"
                        echo "Saving to: ${OUTPUT_FILE}"
                        
                        # Execute the blastn command
			echo $input_fasta
			echo $reference_fasta
                        conda run -n blastn_kmer blastn -query "$input_fasta" \
                               -subject "$reference_fasta" \
                               -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen" \
                               > "$OUTPUT_FILE"
                        
                        if [ $? -ne 0 ]; then
                            echo "!!! WARNING: BLASTN failed for $input_fasta against $reference_fasta."
                        fi
                    fi
                done
            fi
        done
done

echo "--- 6. Running check_classification_parameters.py (File-level Idempotency) ---"

# Regenerate the list of reference basenames used in the previous step.
ALL_REFERENCE_BASENAMES=()
for ref_fasta in "${TARGET_DIR}"/*.fa "${TARGET_DIR}"/*.fna; do
    if [ -f "$ref_fasta" ]; then
        ref_basename=$(basename "$ref_fasta")
        ref_basename="${ref_basename%.*}" # Remove extension
        ALL_REFERENCE_BASENAMES+=("$ref_basename")
    fi
done


# The outer loop iterates over all monitored layers.
for layer in "${LAYERS_TO_MONITOR[@]}"; do
    # The inner loop iterates over all collected reference basenames.
    for reference in "${ALL_REFERENCE_BASENAMES[@]}"; do
        INPUT_FILE="${PCA_DATA_DIR}/pca_data_${layer}.pkl"
        OUTPUT_FILE="${PKL_POSITIVE_CONTIGS}/layer_${layer}_ref_${reference}.pkl" # Output file includes reference
        BLAST_DIR="${BLASTN_DIR}/${SOURCES[1]}" # Hardcoded as per prompt

        # IDEMPOTENCY CHECK (Per Layer/Reference File): Check if the classification output file exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping Classification Check: Output file already exists: $OUTPUT_FILE"
            continue
        fi

        echo "Processing Layer: ${layer} with Reference: ${reference}"
        echo "Input: ${INPUT_FILE} | Output: ${OUTPUT_FILE}"

        # Execute the classification check script
        python check_classification_parameters.py \
            --input_file "$INPUT_FILE" \
            --reference_name "$reference" \
            --blast_dir "$BLAST_DIR" \
            --output_file "$OUTPUT_FILE" \
	    --labels $SOURCES[1] $SOURCES[2] $SOURCES[3]
	echo $SOURCES

        if [ $? -ne 0 ]; then
            echo "!!! WARNING: check_classification_parameters.py failed for layer ${layer} and reference ${reference}."
        fi
    done
done

echo "--- 7. Running summarize_results.py (File-level Idempotency) ---"

LAYERS_STRING=$(IFS=$' '; echo "${LAYERS_TO_MONITOR[*]}")

# Loop over all reference basenames
for reference in "${ALL_REFERENCE_BASENAMES[@]}"; do
    REPORT_PATH="${OUTPUT_REPORT_DIR}/${reference}_summary_report.csv"
    INPUT_DIR=$PKL_POSITIVE_CONTIGS
    
    # IDEMPOTENCY CHECK (Per Reference File): Check if the final report file exists
    if [ -f "$REPORT_PATH" ]; then
        echo "Skipping Summary Report: Output file already exists: $REPORT_PATH"
        continue
    fi

    echo "Summarizing results for Reference: ${reference} across layers..."
    echo "Saving report to: ${REPORT_PATH}"

    # Execute the summary script
    python summarize_results.py \
        --input_dir "$INPUT_DIR" \
        --reference "$reference" \
        --output_path "$REPORT_PATH" \
        --layers $LAYERS_STRING

    if [ $? -ne 0 ]; then
        echo "!!! WARNING: summarize_results.py failed for reference ${reference}."
    fi
done

echo "--- Script finished. ---"
