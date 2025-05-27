"""Script for preprocessing PDB files.

WARNING: NOT TESTED WITH SE(3) DIFFUSION.
This is example code of how to preprocess PDB files.
It does not process extra features that are used in process_pdb_dataset.py.
One can use the logic here to create a version of process_pdb_dataset.py
that works on PDB files.

"""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md


from data import utils as du
from data import errors
from data import parsers


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=50)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='./preprocessed_pdbs')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_file(file_path: str, write_dir: str):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    
    try:
        

        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        os.remove(file_path)
    except Exception as e:
        os.remove(file_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    pdb_dir = args.pdb_dir
    all_file_paths = [
        os.path.join(pdb_dir, x)
        for x in os.listdir(args.pdb_dir) if '.pdb' in x]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)