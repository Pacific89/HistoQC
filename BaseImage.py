import logging
import os
import numpy as np
import inspect
from distutils.util import strtobool
from matplotlib import pyplot as plt
from pathlib import Path

#os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide


def printMaskHelper(type, prev_mask, curr_mask):
    if type == "relative2mask":
        if len(prev_mask.nonzero()[0]) == 0:
            return str(-100)
        else:
            return str(1 - len(curr_mask.nonzero()[0]) / len(prev_mask.nonzero()[0]))
    elif type == "relative2image":
        return str(len(curr_mask.nonzero()[0]) / np.prod(curr_mask.shape))
    elif type == "absolute":
        return str(len(curr_mask.nonzero()[0]))
    else:
        return str(-1)


# this function is seperated out because in the future we hope to have automatic detection of
# magnification if not present in open slide, and/or to confirm openslide base magnification
def getMag(s, params):
    logging.info(f"{s['filename']} - \tgetMag")
    osh = s["os_handle"]
    mag = osh.properties.get("openslide.objective-power", "NA")
    if (
            mag == "NA"):  # openslide doesn't set objective-power for all SVS files: https://github.com/openslide/openslide/issues/247
        mag = osh.properties.get("aperio.AppMag", "NA")
    if (mag == "NA" or strtobool(
            params.get("confirm_base_mag", "False"))):
        # do analysis work here
        logging.warning(f"{s['filename']} - Unknown base magnification for file")
        s["warnings"].append(f"{s['filename']} - Unknown base magnification for file")
    else:
        mag = float(mag)

    return mag


class BaseImage(dict):

    def __init__(self, fname, fname_outdir, params):
        dict.__init__(self)

        self["warnings"] = ['']  # this needs to be first key in case anything else wants to add to it
        self["output"] = []
        self["scan_meta_dict"] = {}
        self["slide_meta_dict"] = {}
        self["wsi_meta_dict"] = {}

        # these 2 need to be first for UI to work
        self.addToPrintList("filename", os.path.basename(fname))
        self.addToPrintList("comments", " ")

        self["outdir"] = fname_outdir
        self["dir"] = os.path.dirname(fname)
        print(fname_outdir)
        print(self["dir"])


        self["os_handle"] = openslide.OpenSlide(fname)
        self["image_base_size"] = self["os_handle"].dimensions
        self["image_work_size"] = params.get("image_work_size", "1.25x")
        self["mask_statistics"] = params.get("mask_statistics", "relative2mask")
        self["base_mag"] = getMag(self, params)
        self.addToPrintList("base_mag", self["base_mag"])

        mask_statistics_types = ["relative2mask", "absolute", "relative2image"]
        if (self["mask_statistics"] not in mask_statistics_types):
            logging.error(
                f"mask_statistic type '{self['mask_statistics']}' is not one of the 3 supported options relative2mask, absolute, relative2image!")
            exit()

        self["img_mask_use"] = np.ones(self.getImgThumb(self["image_work_size"]).shape[0:2], dtype=bool)
        self["img_mask_force"] = []

        self["completed"] = []

    def addToPrintList(self, name, val):
        self[name] = val
        self["output"].append(name)

    def getImgThumb(self, dim):
        key = "img_" + str(dim)
        if key not in self:
            osh = self["os_handle"]
            if dim.replace(".", "0", 1).isdigit(): #check to see if dim is a number
                dim = float(dim)
                if dim < 1 and not dim.is_integer():  # specifying a downscale factor from base
                    new_dim = np.asarray(osh.dimensions) * dim
                    self[key] = np.array(osh.get_thumbnail(new_dim))
                elif dim < 100:  # assume it is a level in the openslide pyramid instead of a direct request
                    dim = int(dim)
                    if dim >= osh.level_count:
                        dim = osh.level_count - 1
                        calling_class = inspect.stack()[1][3]
                        logging.error(
                            f"{self['filename']}: Desired Image Level {dim+1} does not exist! Instead using level {osh.level_count-1}! Downstream output may not be correct")
                        self["warnings"].append(
                            f"Desired Image Level {dim+1} does not exist! Instead using level {osh.level_count-1}! Downstream output may not be correct")
                    logging.info(
                        f"{self['filename']} - \t\tloading image from level {dim} of size {osh.level_dimensions[dim]}")
                    img = osh.read_region((0, 0), dim, osh.level_dimensions[dim])
                    self[key] = np.asarray(img)[:, :, 0:3]
                else:  # assume its an explicit size, *WARNING* this will likely cause different images to have different
                    # perceived magnifications!
                    logging.info(f"{self['filename']} - \t\tcreating image thumb of size {str(dim)}")
                    self[key] = np.array(osh.get_thumbnail((dim, dim)))
            elif "X" in dim.upper():  # specifies a desired operating magnification

                base_mag = self["base_mag"]
                if base_mag != "NA":  # if base magnification is not known, it is set to NA by basic module
                    base_mag = float(base_mag)
                else:  # without knowing base mag, can't use this scaling, push error and exit
                    logging.error(
                        f"{self['filename']}: Has unknown or uncalculated base magnification, cannot specify magnification scale: {base_mag}! Did you try getMag?")
                    return -1

                target_mag = float(dim.upper().split("X")[0])

                down_factor = base_mag / target_mag
                level = osh.get_best_level_for_downsample(down_factor)
                relative_down = down_factor / osh.level_downsamples[level]
                win_size = 2048
                win_size_down = int(win_size * 1 / relative_down)
                dim_base = osh.level_dimensions[0]
                output = []
                for x in range(0, dim_base[0], round(win_size * osh.level_downsamples[level])):
                    row_piece = []
                    for y in range(0, dim_base[1], round(win_size * osh.level_downsamples[level])):
                        aa = osh.read_region((x, y), level, (win_size, win_size))
                        bb = aa.resize((win_size_down, win_size_down))
                        row_piece.append(bb)
                    row_piece = np.concatenate(row_piece, axis=0)[:, :, 0:3]
                    output.append(row_piece)

                output = np.concatenate(output, axis=1)
                output = output[0:round(dim_base[1] * 1 / down_factor), 0:round(dim_base[0] * 1 / down_factor), :]
                self[key] = output
                # if dims are too large, memory errors will arise
                self[key] = self.helper(output)
                self.add_meta_infos()
            else:
                logging.error(
                    f"{self['filename']}: Unknown image level setting: {dim}!")
                return -1




        return self[key]

    def add_meta_infos(self):
        # load values to dictionary
        

        # get total file size (from multiple .dat files)
        
        directory = Path(self['dir'].split('.')[0])
        folder_size = sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())
        single_file_size = Path(self['dir']).stat().st_size
        total_size = folder_size + single_file_size

        compression_quality = self["os_handle"].properties["mirax.LAYER_0_LEVEL_0_SECTION.IMAGE_FORMAT"]
        compression_type = self["os_handle"].properties["mirax.LAYER_0_LEVEL_0_SECTION.IMAGE_COMPRESSION_FACTOR"]
        dim_x = self["os_handle"].dimensions[0]
        dim_y = self["os_handle"].dimensions[1]
        res_x = self["os_handle"].properties["mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X"]
        res_y = self["os_handle"].properties["mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_Y"]
        scan_area_x = float(dim_x) * float(res_x) / 10e3 # res in micro meters 10e-6, divide by 10e3 for mm
        scan_area_y = float(dim_y) * float(res_y) / 10e3 # res in micro meters 10e-6, divide by 10e3 for mm
        # print("scan area x: {0} mm".format(scan_area_x))
        # print("scan area y: {0} mm".format(scan_area_y))


        # SCAN META:
        self["scan_meta_dict"]["scan.resolution"] = str(dim_x) + "," + str(dim_y)
        self["scan_meta_dict"]["scan.resolution-x"] = res_x
        self["scan_meta_dict"]["scan.resolution-y"] = res_y

        self["scan_meta_dict"]["scan.scanner.hw-version"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_SECTION.SCANNER_HARDWARE_VERSION"]
        self["scan_meta_dict"]["scan.scanner.type"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_SECTION.SCANNER_HARDWARE_VERSION"]

        self["scan_meta_dict"]["scan.scanner.manufacturer"] = "3D HISTECH"
        self["scan_meta_dict"]["scan.scanner.serial-number"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_SECTION.SCANNER_HARDWARE_ID"]
        self["scan_meta_dict"]["scan.scanner.sw_version"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_SECTION.SCANNER_SOFTWARE_VERSION"]

        scan_id = self["os_handle"].properties["mirax.GENERAL.SLIDE_ID"]
        self["scan_meta_dict"]["scan.identifier"] = scan_id
        self["scan_meta_dict"]["scan.date"] = self["os_handle"].properties["mirax.GENERAL.SLIDE_CREATIONDATETIME"]
        self["scan_meta_dict"]["scan.compression.type"] = compression_quality
        self["scan_meta_dict"]["scan.compression.quality"] = compression_type
        self["scan_meta_dict"]["scan.area.x"] = scan_area_x
        self["scan_meta_dict"]["scan.area.y"] = scan_area_y
        self["scan_meta_dict"]["scan.area.origin"] = "dicom_top_right"
        self["scan_meta_dict"]["scan.area.bordertop"] = "origin-7mm"
        self["scan_meta_dict"]["scan.area.borderbottom"] = "origin-label_h+76,2mm"
        self["scan_meta_dict"]["scan.area.borderleft"] = "origin+25,4-1mm"
        self["scan_meta_dict"]["scan.area.borderright"] = "origin+1mm"
        self["scan_meta_dict"]["scan.area.maxheight"] = "76,2-label_h-7mm"
        self["scan_meta_dict"]["scan.area.maxwidth"] = "25 -1-1mm"
        # self["meta_info_dict"]["slide_name"] = self["os_handle"].properties["mirax.GENERAL.SLIDE_NAME"]


        # WSI META
        self["wsi_meta_dict"]["wsi.compression.type"] = compression_quality
        self["wsi_meta_dict"]["wsi.compression.quality"] = compression_type
        self["wsi_meta_dict"]["wsi.size"] = total_size

        # SLIDE META
        self["slide_meta_dict"]["slide.identifier.label"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_LEVEL_3_SECTION.BARCODE_VALUE"]
        self["slide_meta_dict"]["slide.label.normalized"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_LEVEL_3_SECTION.BARCODE_VALUE"]
        self["slide_meta_dict"]["slide.label.transcribed"] = self["os_handle"].properties["mirax.NONHIERLAYER_0_LEVEL_3_SECTION.BARCODE_VALUE"]


    def helper(self, output):
        first_layer = output[:,:,0]
        row_sum = np.sum(first_layer, axis=0)
        col_sum = np.sum(first_layer, axis=1)

        x_min = np.argmax(row_sum>0)
        y_min = np.argmax(col_sum>0)
        x_max = len(row_sum) - np.argmax(row_sum[::-1]>0)
        y_max = len(col_sum) - np.argmax(col_sum[::-1]>0)

        # slice array
        output = output[y_min:y_max,x_min:x_max]
        # replace black with white (for later mask calculations)
        output[output == 0] = 255
        return output
