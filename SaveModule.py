import logging
import os
from skimage import io, img_as_ubyte
from distutils.util import strtobool
from skimage import color
import numpy as np
import json

import matplotlib.pyplot as plt


def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out


def saveFinalMask(s, params):
    logging.info(f"{s['filename']} - \tsaveUsableRegion")

    mask = s["img_mask_use"]
    for mask_force in s["img_mask_force"]:
        mask[s[mask_force]] = 0

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_mask_use.png", img_as_ubyte(mask))

    if strtobool(params.get("use_mask", "True")):  # should we create and save the fusion mask?
        img = s.getImgThumb(s["image_work_size"])
        out = blend2Images(img, mask)
        # io.imsave(s["outdir"] + os.sep + s["filename"] + "_fuse.png", img_as_ubyte(out))

    return


def saveThumbnails(s, params):
    logging.info(f"{s['filename']} - \tsaveThumbnail")
    # we create 2 thumbnails for usage in the front end, one relatively small one, and one larger one
    # img = s.getImgThumb(params.get("image_work_size", "1.25x"))
    # io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb.png", img)

    # img = s.getImgThumb(params.get("small_dim", 500))
    # io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb_small.png", img)

    # save additional thumbnails

    basename = s["outdir"] + os.sep + s["filename"]

    label = s["os_handle"].associated_images["label"]
    macro = s["os_handle"].associated_images["macro"]
    thumbnail = s["os_handle"].associated_images["thumbnail"]

    label_resize = label.copy()
    label_resize.thumbnail((500, 500))
    macro_resize = macro.copy()
    macro_resize.thumbnail((500, 500))
    thumbnail_resize = thumbnail.copy()
    thumbnail_resize.thumbnail((500, 500))

    io.imsave(basename + "_macro_small.png", np.asarray(macro_resize))
    io.imsave(basename + "_thumbnail_small.png", np.asarray(thumbnail_resize))
    io.imsave(basename + "_label_small.png", np.asarray(label_resize))

    return

def saveJson(s, params):
    if s["scan_meta_dict"].get("scan.quality.oof-error-rate") == None:
        s["scan_meta_dict"]["scan.quality.oof-error-rate"] = None

    with open(s["outdir"] + os.sep + 'scan_meta.json', 'w') as f:
        json.dump(s["scan_meta_dict"], f)
    with open(s["outdir"] + os.sep + 'slide_meta.json', 'w') as f:
        json.dump(s["slide_meta_dict"], f)
    with open(s["outdir"] + os.sep + 'wsi_meta.json', 'w') as f:
        json.dump(s["wsi_meta_dict"], f)



def save_thumbs(s, params):

    label = s["os_handle"].associated_images["label"]
    macro = s["os_handle"].associated_images["macro"]
    thumbnail = s["os_handle"].associated_images["thumbnail"]

    label.thumbnail((500, 500))
    macro.thumbnail((500,500))
    thumbnail.thumbnail((500,500))

    io.imsave(s["outdir"] + os.sep + s["filename"] + "label_thumb_small.png", label)
    io.imsave(s["outdir"] + os.sep + s["filename"] + "macro_thumb_small.png", macro)
    io.imsave(s["outdir"] + os.sep + s["filename"] + "thumbnail_thumb_small.png", thumbnail)



def renameFolder(s, params):
    # use slide ID from "mirax.GENERAL.SLIDE_ID" from INI file for folder name
    folder_qc_out = os.path.dirname(s["outdir"])
    # folder name with date of output
    folder_name = s["outdir"]

    # folder_to_qc = s["dir"]
    new_name = os.path.join(folder_qc_out, s["scan_meta_dict"]["scan.identifier"])

    os.rename(folder_name, new_name)


