#!/usr/bin/env python3
import datetime
import os
import time
from pathlib import Path
from io import StringIO as IOBuffer
import cv2
from skimage import measure
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Pool
from functools import partial

import plotting
from dcmtools import peek_compressed_study, decompress_case
from dcmtools import peek_study
from dcmtools import load_case, load_study
from dcmtools import clip_voxel_values, resample_volume
import pydicom


def _sdiv(x, y):
    return (x + y - 1) // y


def largest_label_volume(im, background=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != background]
    vals = vals[vals != background]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def preprocessing(voxels, spacing):
    isopencv2 = cv2.__version__.startswith('2.')
    vox = []
    avg_otsu = 0.0
    # slice wise iteration
    # TODO make this faster!
    for idx, s in enumerate(voxels):
        img = s * 255.0
        img = img.astype(np.uint8)

        # Otsu's thresholding
        otsu_th, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        avg_otsu += otsu_th

        # morph operations
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        # compute contours
        if isopencv2:
            _, contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # compute max contour area id
        areas = []
        area_idx = []
        hidx = 0
        while hidx >= 0:
            cnt = contours[hidx]
            area = np.abs(cv2.contourArea(cnt))
            areas.append(area)
            area_idx.append(hidx)
            hidx = hierarchy[0][hidx][0]
        areaidx = np.argmax(areas)
        areaidx = area_idx[areaidx]

        cont = np.zeros_like(img)
        # draw mask of biggest area
        cv2.drawContours(cont, contours, areaidx, (255, 255, 255), -1)
        # filter slice on mask
        masked = np.maximum((255 - cont).astype(np.uint8), (s * 255).astype(np.uint8))
        vox.append(masked)

    avg_otsu /= 1.0 * len(voxels)
    return np.array(vox), avg_otsu


def segment_lung(voxels, threshold):
    thresholded = []
    # slice wise thresholding with global avg. otsu
    for idx, s in enumerate(voxels):
        otsu_th, otsu = cv2.threshold(s, int(threshold), 255, cv2.THRESH_BINARY)
        thresholded.append(255 - otsu)

    thresholded = np.array(thresholded, dtype=np.uint8)
    # compute connected regions
    labels, numlabels = measure.label(thresholded, background=0, return_num=True)
    # get biggest volume
    l_max = largest_label_volume(labels, background=0)
    mask = np.zeros_like(thresholded, dtype=np.uint8)
    if l_max is not None:  # There is at least one ait pocket
        mask[labels == l_max] = 255

    return mask


def crop_border(pixels, tolerance=0):
    """Remove slices from a 3D volume where all pixels < tolerance.

    :param pixels: the 3D volume to crop
    :param tolerance: the tolerance
    :return: cropped 3D volume
    """
    mask = pixels > tolerance
    return pixels[np.ix_(
        mask.any(axis=(1, 2)),
        mask.any(axis=(0, 2)),
        mask.any(axis=(0, 1))
    )]


def execute_directory(input, directory, prefix, csvformat, outfile, output, verbose, debug, seriesiuid):
    logstr = []
    logger = IOBuffer()

    read_start_time = time.time()
    try:
        study_uid = peek_study(input)
    except:
        return

    if csvformat:
        if outfile is not None:
            outfile.close()
        _out_csv = '{}.csv'.format(study_uid)
        # this is needed in every case for images etc.
        folder = prefix / study_uid
        folder.mkdir(exist_ok=True)
        if output:
            _out_csv = folder / _out_csv
        outfile = open(_out_csv, 'w')
        outfile.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                "StudyInstanceUID", "SeriesInstanceUID",
                "PatientSex", "PatientAge", "AcquisitionDate",
                "RawVolume", "AvgHU",
                "DecompressTime", "LuVoXTime", "PlotTime"
            )
        )

    if verbose:
        logstr.append("StudyInstanceUID: {}".format(study_uid))

    slices = load_case(input)

    if verbose:
        logstr.append("Read directory {} with {} slices.".format(input, len(slices)))

    suid, case = load_study(slices)
    if verbose:
        logstr.append("Found {} series with Study UID ({}).".format(len(case), suid))
        logstr.append("Looking for SeriesInstanceUID ...")

    # compute time for decompressing
    read_time = time.time() - read_start_time

    run_on_case(case, logstr, outfile, prefix, seriesiuid, study_uid, suid, read_time, verbose)

    if verbose:
        logstr.append("--- avg. {} seconds ---".format(1.0 * (time.time() - read_start_time) / np.max([1, len(case)])))

    logger.write("\n".join(logstr))
    logger.seek(0)
    return logger


def execute_archive(fp, directory, prefix, csvformat, outfile, output, verbose, debug, seriesiuid, dry=False):
    logstr = []
    logger = IOBuffer()

    tbz_start_time = time.time()

    archive_path = os.path.join(directory, fp)
    try:
        study_uid = peek_compressed_study(archive_path)
    except:
        return

    if csvformat:
        if outfile is not None:
            outfile.close()
        _out_csv = '{}.csv'.format(study_uid)
        # this is needed in every case for images etc.
        folder = prefix / study_uid
        folder.mkdir(exist_ok=True)
        if output:
            _out_csv = folder / _out_csv
        outfile = open(_out_csv, 'w')
        outfile.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                "StudyInstanceUID", "SeriesInstanceUID",
                "PatientSex", "PatientAge", "AcquisitionDate",
                "RawVolume", "AvgHU",
                "DecompressTime", "LuVoXTime", "PlotTime"
            )
        )

    if verbose:
        logstr.append("StudyInstanceUID: {}".format(study_uid))

    slices = decompress_case(archive_path, debug)

    if verbose:
        logstr.append("Decompressed file {} with {} slices.".format(archive_path, len(slices)))

    suid, case = load_study(slices)
    if verbose:
        logstr.append("Found {} series with Study UID ({}).".format(len(case), suid))
        logstr.append("Looking for SeriesInstanceUID ...")

    # compute time for decompressing
    tbz_time = time.time() - tbz_start_time

    run_on_case(case, logstr, outfile, prefix, seriesiuid, study_uid, suid, tbz_time, verbose, dry)

    if verbose:
        logstr.append("--- avg. {} seconds ---".format(1.0 * (time.time() - tbz_start_time) / np.max([1, len(case)])))

    logger.write("\n".join(logstr))
    logger.seek(0)
    return logger


def run_on_case(case, logstr, outfile, prefix, seriesiuid, study_uid, suid, tbz_time, verbose, dry=False):
    # iterate over all series in the study
    for series in case:
        series_start_time = time.time()
        img = series[0]

        if seriesiuid and seriesiuid != img.SeriesInstanceUID:
            logstr.append("Skipping {} due to SeriesInstanceUID filter.".format(img.SeriesInstanceUID))
            continue

        if verbose:
            logstr.append("Found SeriesInstanceUID:")
            logstr.append("{} {} ({}): Dimension ({}/{}/{}) Voxel Spacing ({:.2f}/{:.2f}/{:.2f})".format(
                img.SeriesInstanceUID, img.StudyDescription, img.SeriesDescription,
                img.Rows, img.Columns, len(series),
                img.SliceThicknessX, img.SliceThicknessY, img.SliceThicknessZ
            ))
            logstr.append("#!!# {}, {}, {}, {}".format(img.StudyInstanceUID, img.SeriesInstanceUID, img.KVP, img.Exposure))

        if dry:
            continue

        # untouched, clipped voxel values
        clipped_voxels, spacing = clip_voxel_values(series)

        # resampling
        if resolution is not None:
            if verbose:
                logstr.append("Shape before resampling {}".format(str(clipped_voxels.shape)))

            # resample voxels to given resolution
            clipped_voxels, spacing = resample_volume(clipped_voxels, spacing, resolution, order=1)

            if verbose:
                logstr.append("Shape after resampling", clipped_voxels.shape)

        if verbose:
            logstr.append("Preprozessing voxels...")

        # preprocess voxels
        case_voxels, avg_otsu = preprocessing(clipped_voxels, spacing)

        if verbose:
            logstr.append("Segmenting lung... (threshold = {:.2f} HU)".format(4095.0 * avg_otsu / 255.0 - 1024.0))

        # compute segmentation
        lung_mask = segment_lung(case_voxels, avg_otsu)

        # create output folder
        out_dir = prefix.resolve().absolute() /\
                str(series[0].StudyInstanceUID)
        out_dir.mkdir(parents=True, exist_ok=True)

        # dump bitmask to file
        fpath = prefix.resolve().absolute() / \
                str(img.StudyInstanceUID) / \
                (str(img.SeriesInstanceUID) + ".{}".format("bitmask") + ".np")
        lung_mask.astype(np.bool).tofile(str(fpath.resolve().absolute()))

        # generate a new uid
        uid_gen = pydicom.uid.generate_uid()

        # create DICOM overlays
        for (slice_, m_) in zip(series, lung_mask):
            bits = m_.flatten()

            padded_bits = _sdiv(len(bits), 8) * 8
            if len(bits) < padded_bits:
                bits = np.append(bits, np.zeros((padded_bits - len(bits),), dtype=np.uint8))

            if slice_.is_little_endian:
                bits = np.reshape(bits, (-1, 8))
                bits = np.fliplr(bits)

            bytes_ = np.packbits(bits)

            # set new seriesinstanceuid
            slice_[0x0020, 0x000e] = pydicom.DataElement((0x0020, 0x000e), "UI", uid_gen)
            slice_[0x0008, 0x0018] = pydicom.DataElement((0x0008, 0x0018), "UI", pydicom.uid.generate_uid())

            # update series description
            slice_[0x0008, 0x103e].value = str(slice_[0x0008, 0x103e].value) + " (Volumetrie)"

            # set DICOM elements
            # TODO check if overlay exists
            if (0x6000, 0x3000) in slice_:
                print("Warning: Overwriting existing overlay!")

            overlay_data = pydicom.DataElement((0x6000, 0x3000), "OW", bytes_.tobytes())
            slice_[0x6000, 0x3000] = overlay_data
            overlay_rows = pydicom.DataElement((0x6000, 0x0010), "US", m_.shape[0])
            slice_[0x6000, 0x0010] = overlay_rows
            overlay_cols = pydicom.DataElement((0x6000, 0x0011), "US", m_.shape[1])
            slice_[0x6000, 0x0011] = overlay_cols
            number_of_frames_in_overlay = pydicom.DataElement((0x6000, 0x0015), "IS", 1)
            slice_[0x6000, 0x0015] = number_of_frames_in_overlay
            overlay_description = pydicom.DataElement((0x6000, 0x0022), "LO", "LuVoX Lung Volume Mask")
            slice_[0x6000, 0x0022] = overlay_description
            overlay_label = pydicom.DataElement((0x6000, 0x1500), "LO", "LuVoX Lung Volume Mask")
            slice_[0x6000, 0x1500] = overlay_label
            overlay_subtype = pydicom.DataElement((0x6000, 0x0045), "LO", "AUTOMATED")
            slice_[0x6000, 0x0045] = overlay_subtype
            overlay_type = pydicom.DataElement((0x6000, 0x0040), "CS", "G")
            slice_[0x6000, 0x0040] = overlay_type
            overlay_origin = pydicom.DataElement((0x6000, 0x0050), "SS", [1, 1])
            slice_[0x6000, 0x0050] = overlay_origin
            image_frame_origin = pydicom.DataElement((0x6000, 0x0051), "US", 1)
            slice_[0x6000, 0x0051] = image_frame_origin
            overlay_bits_allocated = pydicom.DataElement((0x6000, 0x0100), "US", 1)
            slice_[0x6000, 0x0100] = overlay_bits_allocated
            overlay_bit_position = pydicom.DataElement((0x6000, 0x0102), "US", 0)
            slice_[0x6000, 0x0102] = overlay_bit_position
            fpath = prefix.resolve().absolute() /\
                    str(slice_.StudyInstanceUID) /\
                    (str(slice_.SeriesInstanceUID)
                     + "_{}".format(str("".join(map(str, slice_.ImagePositionPatient))))
                     + ".dcm")
            slice_.save_as(str(fpath.resolve().absolute()))

        voxelvol = np.prod(spacing)

        if verbose:
            logstr.append("Voxel volume is {}".format(voxelvol))

        mask_sum = np.sum(lung_mask.astype(np.bool))
        raw_volume = mask_sum * voxelvol * 1e-6  # convert to Liter (from Milliliter)
        raw_avg_hu = 4095.0 * np.mean(case_voxels[lung_mask > 0]) / 255.0 - 1024.0

        if verbose:
            logstr.append("## LUNG: #{}; Volume: {:.2f} L; avg HU: {:.2f}".format(mask_sum, raw_volume, raw_avg_hu))

        # time per series
        luvo_time = time.time() - series_start_time
        plot_start_time = time.time()

        # create a pdf for this series
        with PdfPages("{}.pdf".format(prefix / study_uid / img.SeriesInstanceUID)) as pdf:
            # write meta data
            plt.figure(figsize=(8, 6))
            plt.title("LuVoX - Fully automated lung volumetry from CT scans.")
            res_text = "StudyUID: {}\n SeriesUID: {}\n PatientSex: {}\n PatientAge: {}\n AcquisitionDate: {}\n" \
                       " RAW-Volume: {}\n Avg.HU: {}\n Load-time: {}\n LuVoX-time: {}".format(
                img.StudyInstanceUID, img.SeriesInstanceUID,
                img.PatientSex, int(img.PatientAge[:-1]), img.AcquisitionDate,
                raw_volume, raw_avg_hu,
                tbz_time, luvo_time
            )
            plt.text(0.5, 0.5, res_text, horizontalalignment="center", verticalalignment="center")
            plt.axis("off")
            pdf.savefig()
            plt.close()

            # plot RAW data
            twodplotz = plotting.plot_2d(clipped_voxels)
            for desc in twodplotz:
                plt.figure(figsize=(8, 6))
                plt.suptitle("SeriesInstanceUID {}".format(img.SeriesInstanceUID))
                plt.title("RAW-{}".format(desc))
                plot = twodplotz[desc]
                plt.imshow(plot)
                pdf.savefig()
                plt.close()

            # plot preprocessed voxels
            twodplotz = plotting.plot_2d(case_voxels)
            for desc in twodplotz:
                plt.figure(figsize=(8, 6))
                plt.suptitle("SeriesInstanceUID {}".format(img.SeriesInstanceUID))
                plt.title("Preprocessed-{}".format(desc))
                plot = twodplotz[desc]
                plt.imshow(plot)
                pdf.savefig()
                plt.close()

            # plot lung mask
            twodplotz = plotting.plot_2d(lung_mask)
            for desc in twodplotz:
                plt.figure(figsize=(8, 6))
                plt.suptitle("SeriesInstanceUID {}".format(img.SeriesInstanceUID))
                plt.title("Mask-{}".format(desc))
                plot = twodplotz[desc]
                plt.imshow(plot)
                pdf.savefig()
                plt.close()

            overlays = plotting.create_overlays(clipped_voxels, lung_mask)
            for idx, ov in enumerate(overlays):
                plt.figure(figsize=(8, 6))
                plt.suptitle("SeriesInstanceUID {}".format(img.SeriesInstanceUID))
                plt.title("Overlay-{}".format(idx))
                plt.imshow(ov)
                pdf.savefig()
                plt.close()
                matplotlib.pyplot.imsave(
                    str(prefix / study_uid / "{}-{}-{}.{}".format(img.SeriesInstanceUID, "overlay", idx, "png")),
                    ov
                )

        plot_time = time.time() - plot_start_time
        result = "{},{},{},{},{},{},{},{},{},{}".format(
            suid, img.SeriesInstanceUID,
            img.PatientSex, int(img.PatientAge[:-1]), img.AcquisitionDate,
            raw_volume, raw_avg_hu,
            tbz_time, luvo_time, plot_time
        )

        logstr.append(result)

        outfile.write(result)
        outfile.write("\n")
        outfile.flush()


def process_archives(tbz_list, prefix, outfile, args):
    """Processes a list of tbz archives

    :param tbz_list: list of filenames
    :param prefix: output prefix
    :param outfile: output filename
    :param args: dict of various arguments
    :return: list of strings containing lines of logs
    """
    if args.verbose:
        print("Found {} .tbz files.".format(len(tbz_list)))
    _logs = []
    with tqdm(desc="LuVoX", total=len(tbz_list), unit="file") as pbar:
        if len(tbz_list) == 1:
            _log = execute_archive(tbz_list[0],
                            prefix=prefix, directory=args.input,
                            csvformat=args.csvformat, outfile=outfile,
                            output=args.output, verbose=args.verbose,
                            debug=args.debug, seriesiuid=args.seriesiuid, dry=args.dry)
            pbar.update(1)
            _logs.append(_log)
            return _logs

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            for _log in pool.imap_unordered(
                    partial(
                        execute_archive,
                        prefix=prefix, directory=args.input,
                        csvformat=args.csvformat, outfile=outfile,
                        output=args.output, verbose=args.verbose,
                        debug=args.debug, seriesiuid=args.seriesiuid
                    ),
                    tbz_list
            ):
                pbar.update(1)
                _logs.append(_log)
    return _logs


def process_single_folder(input, prefix, outfile, args):
    """Process a DICOM folder

    :param input: path to directory
    :param prefix: output prefix
    :param outfile: output filename
    :param args: dict of various arguments
    :return: list of strings containing lines of logs
    """
    _logs = []
    _log = execute_directory(input,
                             prefix=prefix, directory=args.input,
                             csvformat=args.csvformat, outfile=outfile,
                             output=args.output, verbose=args.verbose,
                             debug=args.debug, seriesiuid=args.seriesiuid)
    _logs.append(_log)
    return _logs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="LuVoX",
        description="Workflow-centred open-source fully automated lung volumetry in chest CT."
                    " https://doi.org/10.1016/j.crad.2019.08.010")
    parser.add_argument("--seriesiuid",
                        help="Only process given SeriesInstanceUID.")
    parser.add_argument("--studyiuid",
                        help="Only process given StudyInstanceUID.")
    parser.add_argument("--res",
                        help="Equal resolution for all dimensions.")
    parser.add_argument("--prefix", dest="output",
                        help="Output directory prefix for things.")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enable debug mode.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Show verbose output.")
    parser.add_argument("--plot",  default=False, action="store_true",
                        help="Save plots.")
    parser.add_argument("--plot3d",  default=False, action="store_true",
                        help="Save 3D plots.")
    parser.add_argument("--csvformat", default=False, action="store_true",
                        help="Enable CSV output format.")
    parser.add_argument("--dry", default=False, action="store_true",
                        help="Dry run.")
    parser.add_argument("input", type=str,
                        help="Path to DICOM tbz archive, directory with tbz archives, or DICOM directory.")

    args = parser.parse_args()

    # resample resolution
    resolution = None
    if args.res:
        resolution = np.array(3 * [args.res]).astype(np.float32)

    # create list of input files/dirs
    tbz_list = []
    mhd_list = []
    if os.path.isdir(args.input):
        tbz_list = [f for f in os.listdir(args.input) if f.endswith('.tbz')]
    elif os.path.isfile(args.input) and args.input.endswith('.tbz'):
        tbz_list = [args.input]
    elif os.path.isfile(args.input) and args.input.endswith('.mhd'):
        # here we have a MHD file!
        mhd_list = [args.input]
    else:
        raise RuntimeError("No input files!")

    prefix = Path(".")
    if args.output:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        prefix = Path(args.output)

        if args.verbose:
            print("Output prefix is '{}'".format(prefix))

    outfile = None
    if not args.csvformat:
        out_csv = '{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        if args.output:
            out_csv = prefix / out_csv

        outfile = open(out_csv, 'w')

    outfile.write("StudyInstanceUID,SeriesInstanceUID,PatientSex,"
                  "PatientAge,AcquisitionDate,Volume,AvgHU,ReadTime,LuVoXTime,PlotTime")
    outfile.write("\n")

    logs = []
    if tbz_list:
        # run on a list of archives
        logs = process_archives(tbz_list, prefix, outfile, args)
    elif mhd_list:
        # TODO: Currently not needed
        pass
    else:
        if os.path.isdir(args.input):
            # run on a single DICOM directory
            logs = process_single_folder(args.input, prefix, outfile, args)
        else:
            raise RuntimeError("Input is not supported!")

    print()
    # print all the logs
    for l in logs:
        print(l.getvalue())
