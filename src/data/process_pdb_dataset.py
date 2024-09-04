"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""
import sys
sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')
import argparse
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import time

import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser
from tqdm import tqdm

from data import errors, mmcif_parsing, parsers
from data import utils as du

# Define the parser
parser = argparse.ArgumentParser(description='mmCIF processing script.')
parser.add_argument('--mmcif_dir',help='Path to directory with mmcif files.',type=str)
parser.add_argument('--max_file_size',help='Max file size.',type=int,default=3000000)  # Only process files up to 3MB large.
parser.add_argument('--min_file_size',help='Min file size.',type=int,default=1000)  # Files must be at least 1KB.
parser.add_argument('--max_resolution',help='Max resolution of files.',type=float,default=5.0)
parser.add_argument('--max_len',help='Max length of protein.',type=int,default=512)
parser.add_argument('--num_processes',help='Number of processes.',type=int,default=100)
parser.add_argument('--write_dir',help='Path to write results to.',type=str,default='./data/processed_pdb')
parser.add_argument('--debug',help='Turn on for debugging.',action='store_true')
parser.add_argument('--verbose',help='Whether to log everything.',action='store_true')


def _retrieve_mmcif_files(
        mmcif_dir: str, max_file_size: int, min_file_size: int, debug: bool):
    """Set up all the mmcif files to read."""
    print('Gathering mmCIF paths')
    total_num_files = 0
    all_mmcif_paths = []
    for subdir in tqdm(os.listdir(mmcif_dir)):
        mmcif_file_dir = os.path.join(mmcif_dir, subdir)
        if not os.path.isdir(mmcif_file_dir):
            continue
        for mmcif_file in os.listdir(mmcif_file_dir):
            if not mmcif_file.endswith('.cif'):
                continue
            mmcif_path = os.path.join(mmcif_file_dir, mmcif_file)
            total_num_files += 1
            if min_file_size <= os.path.getsize(mmcif_path) <= max_file_size:
                all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 100:
            # Don't process all files for debugging
            break
    print(
        f'Processing {len(all_mmcif_paths)} files our of {total_num_files}')
    return all_mmcif_paths


def process_mmcif(
        mmcif_path: str, max_resolution: int, max_len: int, write_dir: str):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '')
    metadata['pdb_name'] = mmcif_name
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f'{mmcif_name}.pkl')
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    metadata['processed_path'] = processed_mmcif_path
    try:
        with open(mmcif_path, 'r') as f:
            parsed_mmcif = mmcif_parsing.parse(
                file_id=mmcif_name, mmcif_string=f.read())
    except:
        raise errors.FileExistsError(
            f'Error file do not exist {mmcif_path}'
        )
    metadata['raw_path'] = mmcif_path
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ','.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ','.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata['oligomeric_count'] = oligomeric_count
    metadata['oligomeric_detail'] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    metadata['resolution'] = mmcif_resolution
    metadata['structure_method'] = mmcif_header['structure_method']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}')

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in parsed_mmcif.structure.get_chains()}
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
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['seq_len'] = len(complex_aatype)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    if complex_aatype.shape[0] > max_len:
        raise errors.LengthError(
            f'Too long {complex_aatype.shape[0]}')

    try:
        
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMCIFParser()
        struc = p.get_structure("", mmcif_path)
        io = PDBIO()
        io.set_structure(struc)
        pdb_path = mmcif_path.replace('.cif', '.pdb')
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        os.remove(pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]

    # Write features to pickles.
    du.write_pkl(processed_mmcif_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(
        all_mmcif_paths, max_resolution, max_len, write_dir):
    all_metadata = []
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            metadata = process_mmcif(
                mmcif_path,
                max_resolution,
                max_len,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {mmcif_path}: {e}')
    return all_metadata


def process_fn(
        mmcif_path,
        verbose=None,
        max_resolution=None,
        max_len=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_mmcif(
            mmcif_path,
            max_resolution,
            max_len,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {mmcif_path}: {e}')


def main(args):
    # Get all mmcif files to read.
    all_mmcif_paths = _retrieve_mmcif_files(
        args.mmcif_dir, args.max_file_size, args.min_file_size, args.debug)
    print(all_mmcif_paths)
    total_num_paths = len(all_mmcif_paths)
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
            all_mmcif_paths,
            args.max_resolution,
            args.max_len,
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            max_resolution=args.max_resolution,
            max_len=args.max_len,
            write_dir=write_dir)
        # Uses max number of available cores.
        with mp.Pool(processes=1) as pool:
            all_metadata = pool.map(_process_fn, all_mmcif_paths)
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