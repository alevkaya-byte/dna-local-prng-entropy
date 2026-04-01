# DNA-Local PRNG for Entropy

Code and reproducibility materials for the Entropy manuscript on a DNA-local dual-head Transformer-based pseudo-random generation framework.

## Overview

This repository contains the main code and representative output files used in the manuscript on DNA-local pseudo-random generation with a dual-head Transformer architecture.

The study includes two main code tracks:

1. **Trained regime (`kit_real.py`)**
   - used for generation under trained settings
   - supports real-data and synthetic-data based workflows

2. **No-ref / no-train regime (`kit_noref.py`)**
   - used for data-independent generation
   - supports reference-free output generation without data-driven training

## Repository Contents

- `kit_real.py`  
  Main code for the trained generation pipeline.

- `kit_noref.py`  
  Main code for the no-ref / no-train generation pipeline.

- Representative output files  
  Example output files from the experimental regimes are included in this repository. These include:
  - bitstream output files (`*.bits.txt`)
  - DNA sequence output files (`*.dna.txt`)
  - metadata files (`*.json`)
  - rule trace files (`*.rules.txt`)

## Output File Types

Each generation run may produce the following files:

- `*.bits.txt`  
  Generated bitstream output.

- `*.dna.txt`  
  Generated DNA sequence output.

- `*.json`  
  Run metadata such as seed, output length, rule counts, sampling settings, smoothing settings, and performance statistics.

- `*.rules.txt`  
  Rule trace showing dynamic DNA-to-bit mapping selections during generation.

## Experimental Regimes

The manuscript evaluates the framework under three regimes:

- **Real-data trained regime**
- **Synthetic-data trained regime**
- **No-ref / no-train regime**

Representative outputs from these regimes are included here for reproducibility demonstration.

## Reproducibility Note

This repository currently includes the main code files and representative example outputs.  
The full collection of independent streams, complete seed tables, and extended reproducibility materials may be released in a later archival version of the repository or supplementary archive.

## Basic Usage

### Trained pipeline

Example commands from `kit_real.py`:

```bash
python kit_real.py train --real real_data_100k.txt --ckpt checkpoints/dna_prng.pt
python kit_real.py gen --ckpt checkpoints/dna_prng.pt --real real_data_100k.txt --bits 1000000 --seed 12345 --out outputs/bits_S12345.txt
python kit_real.py gen-many --ckpt checkpoints/dna_prng.pt --real real_data_100k.txt --bits 1000000 --count 100 --seed-base 1000 --out-dir outputs/streams



