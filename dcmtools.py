#!/usr/bin/env python3
"""DCMTools for loading (compressed) DICOM studies and series.

This module provides various methods to load compressed archives or
a single directory, which can contain (multiple) DICOM studies / series.

"""
from __future__ import print_function
import tarfile
import os
import time

try:
    # Python 2 compatibility
    from StringIO import StringIO as IOBuffer
except ImportError:
    from io import BytesIO as IOBuffer

import pydicom
import numpy as np
import scipy.ndimage.interpolation


def peek_study(path):
    """Peeks the `StudyInstanceUID` of a DICOM directory, that is,
     reads the first DICOM file in that directory and returns
     the associated ID.

    Args:
        path (str): The directory which contains at least one DICOM file.

    Returns:
        str or None: The StudyInstanceUID if possible, None otherwise.

    """
    dcm = None
    for s in os.listdir(path):
        try:
            dcm = pydicom.read_file(path + '/' + s)
            break
        except:
            continue
    if dcm is not None:
        return dcm.StudyInstanceUID
    return None


def peek_compressed_study(archive_path):
    """Peeks the `StudyInstanceUID` of a compressed DICOM archive, that is,
    reads the first DICOM file in the archive and returns the associated ID.

    Args:
        archive_path (str): The filepath of the archive.

    Returns:
        str or None: The StudyInstanceUID if possible, None otherwise.

    """
    mode = ''
    if archive_path.lower().endswith('.gz') or archive_path.lower().endswith('.tgz'):
        mode = 'r:gz'
    elif archive_path.lower().endswith('.bz2') or archive_path.lower().endswith('.tbz'):
        mode = 'r:bz2'
    try:
        tar_archive = tarfile.open(archive_path, mode)
    except:
        return None
    dcm = None
    while True:
        tarinfo = tar_archive.next()
        if tarinfo is None:
            break
        if tarinfo.isfile():
            file_object = tar_archive.extractfile(tarinfo)
            file_like_object = IOBuffer(file_object.read())
            file_object.close()
            file_like_object.seek(0)
            try:
                dcm = pydicom.read_file(file_like_object)
            except:
                continue
            break
    if dcm is not None:
        return dcm.StudyInstanceUID
    return None


def load_study(slices, debug=False):
    """Loads all DICOM files in the given list of filepaths

    Args:
        slices (list): The filepath of the archive.
        debug (bool): To print debug information.

    Returns:
        study_uid: The StudyInstanceUID of the study.
        clean_series: A list of all DICOM images.

    """
    # remove sr and annotation files
    # only kep image files
    clean_slices = []
    for s in slices:
        try:
            # try to get the position, fails for SR files and other which have no image content
            ipp = s.ImagePositionPatient[2]
            clean_slices.append(s)
        except:
            if debug:
                print("Removing slice PatientID:{}; SOPClassUID:{}".format(s.PatientID, s.SOPClassUID))
    slices = clean_slices
    del clean_slices  # free some memory

    # find all series in this DICOM directory
    slices.sort(key=lambda x: x.SeriesInstanceUID)
    current = None
    cidx = -1
    series = []
    for s in slices:
        if not current == s.SeriesInstanceUID:
            current = s.SeriesInstanceUID
            cidx += 1
            series.append([])
        series[cidx].append(s)
    # sort every series by z coordinate
    for s in series:
        s.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    study_uid = None
    clean_series = []
    # let's print some information about the series
    clean_idx = 0
    for idx, series_no in enumerate(series):
        img = series_no[0]
        if len(series_no) > 1:
            snd = series_no[1]
            thickness_x = img.PixelSpacing[0]
            thickness_y = img.PixelSpacing[1]
            thickness_z = np.abs(img.ImagePositionPatient[2] - snd.ImagePositionPatient[2])
        else:
            # continue of only one slice in series
            continue

        # continue if slices have no thickness
        if np.max([thickness_x, thickness_y, thickness_z]) <= 0.0:
            continue

        # try:
        #     img.SpacingBetweenSlices
        # except:
        #     # continue if there is no spacing
        #     continue

        image_types = img.ImageType
        if not image_types[0] == 'ORIGINAL' and image_types[1] == 'PRIMARY':
            # is the image an ORIGINAL Image; an image whose pixel values are based on original or source data
            # is the image a PRIMARY Image; an image created as a direct result of the Patient examination
            continue

        for slice_no in series_no:
            slice_no.SliceThicknessX = thickness_x
            slice_no.SliceThicknessY = thickness_y
            slice_no.SliceThicknessZ = thickness_z

        # series is fine, use it!
        clean_series.append(series_no)
        clean_idx += 1

        if study_uid != img.StudyInstanceUID and study_uid is not None:
            if debug:
                print("Warning: Found multiple Study Instance UIDs in directory!")

        if study_uid is None:
            study_uid = img.StudyInstanceUID

    return study_uid, clean_series


def clip_voxel_values(slices):
    """Clips the values of the given list of DICOM images to be in range [0,1),
    that is, the values are scaled to be in range [0,1), all values below 0 are clipped to 0
    and all values above 1 are clipped to 1.
    The result is returned as a float array.

    Args:
        slices (list): A list of DICOM images.

    Returns:
        slices (numpy.array): The convertes images as a numpy array
        spacing (numpy.array): The spacing of a single slice

    """
    image = np.stack([s.pixel_array for s in slices])
    # Convert to float32 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.float32)

    # Convert to Hounsfield units (HU)
    # Apply linear transformation from disk rep to memory rep
    # See https://stackoverflow.com/questions/10193971/rescale-slope-and-rescale-intercept
    # And https://stackoverflow.com/questions/8756096/window-width-and-center-calculation-of-dicom-image/8765366#8765366
    for idx, slice in enumerate(slices):
        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope

        img = image[idx]
        img[img < 0] = 0
        img[img > 4095] = 4095

        if slope != 1:
            print("WARNING:", "DICOM RescaleSlope != 1", "We should probably do something here.")
            # img = slope * slice.astype(np.float64)
            # img = slice.astype(np.int16)

        # memory rep is in [0,4096)
        # so no need to shift data
        # img += np.int16(intercept)

        # divide with max value, so that values are in range [0,1)
        img /= np.float32(4095)

        image[idx] = img

    im = slices[0]
    return np.array(image, dtype=np.float32),\
           np.array(
               [im.SliceThicknessX, im.SliceThicknessY, im.SliceThicknessZ],
               dtype=np.float32)


def resample_volume(vol, spacing, new_spacing, order=2):
    """Resample the given volume with the given spacing to the given new_spacing.

    Args:
        vol (numpy.array): The 3D volume as numpy array in `z` `x` `y`
        spacing (numpy.array): The spacing of the 3D volume in `x` `y` `z`
        new_spacing (numpy.array): The new spacing to resample the volume to in `x` `y` `z`
        order (int): The order of the spline interpolation

    Returns:
        volume (numpy.array): The resampled 3D volume
        spacing (numpy.array): The spacing of the resampled 3D volume in `x` `y` `z`

    """
    # volume has order z x y, so roll things around
    vol_shape = np.roll(vol.shape, -1)
    new_shape = np.round(np.multiply(vol_shape, np.divide(spacing, new_spacing)))
    true_spacing = np.multiply(spacing, np.divide(vol_shape, new_shape))
    resize_factor = np.divide(new_shape, vol_shape)
    vol = scipy.ndimage.interpolation.zoom(
        vol, resize_factor, mode='nearest', order=order)
    return vol, true_spacing


def decompress_case(archive_path, debug=False):
    """Decompress a archive in memory.

    Args:
        archive_path (str): The filepath of the archive
        debug (bool): To print debug information

    Returns:
        slices (list): The list of decompressed files

    """
    mode = ''
    if archive_path.lower().endswith('.gz') or archive_path.lower().endswith('.tgz'):
        mode = 'r:gz'
    elif archive_path.lower().endswith('.bz2') or archive_path.lower().endswith('.tbz'):
        mode = 'r:bz2'
    tar_archive = tarfile.open(archive_path, mode)
    slices = []
    if debug:
        print("Untar...")

    start = time.time()

    while True:
        tarinfo = tar_archive.next()
        if tarinfo is None:
            break
        if tarinfo.isfile():
            file_object = tar_archive.extractfile(tarinfo)
            file_like_object = IOBuffer(file_object.read())
            file_object.close()
            file_like_object.seek(0)
            try:
                dcm = pydicom.read_file(file_like_object)
                slices.append(dcm)
            except:
                continue

    end_untar = time.time()
    if debug:
        print("Untar time: {}".format(end_untar - start))
    return slices


def load_case(path):
    """Loads all DICOM files in the given directory.

    Args:
        path (str): The filepath of the directory

    Returns:
        slices (list): The list of loaded DICOM files

    """
    slices = []
    for s in os.listdir(path):
        try:
            dcm = pydicom.read_file(path + '/' + s)
            slices.append(dcm)
        except:
            continue
    return slices


if __name__ == '__main__':
    """Example usage of DCMTools functions.

    This method demonstrates the usage of the provided functions to peek and load
    compressed ('gz', 'tgz', 'bz2', 'tbz') archives or decompressed DICOM files
    in a directory.

    Example:
        See `--help` for more information on how to use this example:

            $ python3 dcmtools.py --help

    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="DCMTools", description="Loads DICOM files from directories and archives.")

    # need to be
    parser.add_argument("input", help="input file")

    parser.add_argument('--resolution', default=[1.0, 1.0, 1.0], nargs=3, metavar=('x', 'y', 'z'),
                        type=float, help='resampled resolution in x y z')
    parser.add_argument("--output", type=str, help="save the output file")
    parser.add_argument("--studyuid", type=str, help="filter by study uid")
    parser.add_argument("--seriesuid", type=str, help="filter by series uid")
    parser.add_argument("--debug", action='store_true',
                        default=False, help="print debug information")
    args = parser.parse_args()

    # print the usage
    parser.print_usage()

    # transform input variables
    resolution = np.array(args.resolution)
    p = Path(args.input)

    archives = []
    archive_suffixes = ['gz', 'tgz', 'bz2', 'tbz']
    if p.is_dir():
        # try to find a file with archive suffix in the folder
        try:
            archives = [item for sblst in [list(p.glob('*.' + sffx)) for sffx in archive_suffixes] for item in sblst]
        except:
            pass

    # check if input file is already a compressed archive
    if p.exists() and not p.is_dir():
        # it's a file!
        if p.suffix[1:] in archive_suffixes:
            archives.append(p)

    # peek all study id's
    for idx, arch in enumerate(archives):
        arch_path = str(arch)
        print("Found a *.tbz archive:", arch_path)
        study_uid = peek_compressed_study(arch_path)
        print("StudyInstanceUID: {}".format(study_uid))

    # if at least one file is found,
    # we can decompress it and get suid and slice count
    if len(archives):
        arch_path = str(archives[0])
        print("Found a *.tbz archive:", arch_path)
        study_uid = peek_compressed_study(arch_path)
        print("StudyInstanceUID: {}".format(study_uid))
        dcm_slices = decompress_case(arch_path)
        print("Decompressed file {} with {} slices.".format(arch_path, len(dcm_slices)))
        exit(0)

    # no compressed archives found
    # so proved input must be a directory to decompressed files
    case_path = str(p)
    print("Found DICOM directory:", case_path)
    # peek directory to get study id
    study_uid = peek_study(case_path)
    print("StudyInstanceUID: {}".format(study_uid))
    dcm_slices = load_case(case_path)
    study_uid, clean_series = load_study(dcm_slices)
    print("Found {} series with study uid ({}) in directory '{}':".format(len(clean_series), study_uid, case_path))
    series = None
    for series in clean_series:
        img = series[0]
        print("{} {} ({}): Dimension ({}/{}/{}) Thickness ({}/{}/{}:{}) ImageType ({}) SOPClassUID ({})".format(
            img.SeriesInstanceUID, img.StudyDescription, img.SeriesDescription,
            img.Rows, img.Columns, len(series),
            img.SliceThicknessX, img.SliceThicknessY, img.SliceThicknessZ,
            img.SpacingBetweenSlices,
            img.ImageType, img.SOPClassUID
        ))
    if series:
        print("Load last seen series...")
        # untouched voxel values
        case_voxels, spacing = clip_voxel_values(series)
        # resampled
        resampled, data_spacing = resample_volume(case_voxels, spacing, resolution, order=1)
        print("Done!")
