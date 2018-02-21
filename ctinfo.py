#!/usr/bin/env python3
"""CTInfo

This module provides various methods to extract meta information
from DICOM files or series in directories and archives.
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import shutil
import scipy.misc
import numpy as np
import pandas as pd
import multiprocessing
import dcmtools
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
try:
    # Python 2 compatibility
    from StringIO import StringIO as IOBuffer
except ImportError:
    from io import StringIO as IOBuffer


def process_archive(archive_filepath, prefix=".", img_format="jpg"):
    logger = IOBuffer()
    archive_path = str(archive_filepath)

    case = dcmtools.decompress_case(archive_path)
    studyInstanceUID, study = dcmtools.load_study(case)

    if studyInstanceUID is None:
        logger.write('No StudyInstanceUID found in filename {}'.format(archive_path))
        return

    folder = Path(prefix) / studyInstanceUID
    folder.mkdir(exist_ok=True)

    logger.write("Saving meta info: {}".format(folder))

    studies_meta = []

    for series in study:
        img = series[0]

        meta_info = {
            'Filename': str(archive_path),
            'StudyInstanceUID': str(img.StudyInstanceUID),
            'SeriesInstanceUID': str(img.SeriesInstanceUID),
            'PatientSex': str(img.PatientSex),
            'AcquisitionDate': int(img.AcquisitionDate),
            'SliceThicknessX': float(img.SliceThicknessX),
            'SliceThicknessY': float(img.SliceThicknessY),
            'SliceThicknessZ': float(img.SliceThicknessZ),
            'PatientAge': int(img.PatientAge[:-1]),
            'StudyDescription': str(img.StudyDescription),
            'SeriesDescription': str(img.SeriesDescription)
        }
        studies_meta.append(meta_info)

        logger.write(str(meta_info))

        voxels, spacing = dcmtools.clip_voxel_values(series)

        mid_slice_X = voxels[int(voxels.shape[0]//2), :, :]
        mid_slice_Y = np.flipud(voxels[:, int(voxels.shape[1]//2), :])
        mid_slice_Z = np.flipud(voxels[:, :, int(voxels.shape[2]//2)])

        scipy.misc.imsave(
            folder / "{}-{}.{}".format(img.SeriesInstanceUID, "X", img_format),
            mid_slice_X)
        scipy.misc.imsave(
            folder / "{}-{}.{}".format(img.SeriesInstanceUID, "Y", img_format),
            mid_slice_Y)
        scipy.misc.imsave(
            folder / "{}-{}.{}".format(img.SeriesInstanceUID, "Z", img_format),
            mid_slice_Z)

    df_meta_info = pd.DataFrame(studies_meta)
    df_meta_info.to_csv(folder / "{}.{}".format(studyInstanceUID, "csv"))

    # rewind the tape
    logger.seek(0)
    return logger


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(
        prog="CTInfo",
        add_help="Extracts meta information from DICOM files or series in compressed archives.")
    parser.add_argument("input", help="input file or directory")
    parser.add_argument("--prefix", default=".", help="output prefix path")
    args = parser.parse_args()

    # whatever one provides as input
    # in the end only one file is used to extract meta information

    # declare some variables
    archive_suffixes = ['gz', 'tgz', 'bz2', 'tbz']

    path = Path(args.input)

    # sanity check, raise error if input doesn't exist
    if not path.exists():
        raise ValueError("The input file does not exist.")

    # a list of all files to be processed
    archives_todo = []
    # traverse directory and find archives
    if path.is_dir():
        # grab the first DICOM image in the directory
        for (dirpath, dirnames, filenames) in os.walk(str(path)):
            for fname in filenames:
                fp = Path(dirpath) / fname
                if fp.suffix[1:] in archive_suffixes:
                    archives_todo.append(fp)
    elif path.is_file():
        # only one file given as input
        if path.suffix[1:] in archive_suffixes:
            archives_todo.append(path)

    logger = IOBuffer()
    # start worker processes
    with tqdm(desc="CTInfo", total=len(archives_todo), unit="file") as pbar:
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            for log in tqdm(pool.imap_unordered(partial(process_archive, prefix=args.prefix), archives_todo)):
                pbar.update(1)
                logger.write(log.getvalue())

    log_filep = Path(args.prefix) / "{}.log".format("MetaInfo")
    with open(log_filep, "w") as fp:
        # rewind the tape
        logger.seek(0)
        shutil.copyfileobj(logger, fp)
