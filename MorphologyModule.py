import logging
import os
import numpy as np
from BaseImage import printMaskHelper
from skimage import io, morphology, img_as_ubyte, measure, color, draw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt  # these 2 are used for debugging
from SaveModule import blend2Images #for easier debugging


def removeSmallObjects(s, params):
    logging.info(f"{s['filename']} - \tremoveSmallObjects")
    min_size = int(params.get("min_size", 64))
    img_reduced = morphology.remove_small_objects(s["img_mask_use"], min_size=min_size)
    img_small = np.invert(img_reduced) & s["img_mask_use"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_small_remove.png", img_as_ubyte(img_small))
    s["img_mask_small_filled"] = (img_small * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = img_reduced

    s.addToPrintList("percent_small_tissue_removed",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))


    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.removeSmallObjects: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.removeSmallObjects: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return


def remove_large_objects(img, max_size):
    # code taken from morphology.remove_small_holes, except switched < with >
    selem = ndi.generate_binary_structure(img.ndim, 1)
    ccs = np.zeros_like(img, dtype=np.int32)
    ndi.label(img, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    img_out = img.copy()
    img_out[too_big_mask] = 0
    return img_out


def removeFatlikeTissue(s, params):
    logging.info(f"{s['filename']} - \tremoveFatlikeTissue")
    fat_cell_size = int(params.get("fat_cell_size", 64))
    kernel_size = int(params.get("kernel_size", 3))
    max_keep_size = int(params.get("max_keep_size", 1000))

    img_reduced = morphology.remove_small_holes(s["img_mask_use"], min_size=fat_cell_size)
    img_small = img_reduced & np.invert(s["img_mask_use"])
    img_small = ~morphology.remove_small_holes(~img_small, min_size=9)

    mask_dilate = morphology.dilation(img_small, selem=np.ones((kernel_size, kernel_size)))
    mask_dilate_removed = remove_large_objects(mask_dilate, max_keep_size)

    mask_fat = mask_dilate & ~mask_dilate_removed

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_fatlike.png", img_as_ubyte(mask_fat))
    s["img_mask_fatlike"] = (mask_fat * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = prev_mask & ~mask_fat

    s.addToPrintList("percent_fatlike_tissue_removed",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.removeFatlikeTissue: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.removeFatlikeTissue: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")


def fillSmallHoles(s, params):
    logging.info(f"{s['filename']} - \tfillSmallHoles")
    min_size = int(params.get("min_size", 64))
    img_reduced = morphology.remove_small_holes(s["img_mask_use"], min_size=min_size)
    img_small = img_reduced & np.invert(s["img_mask_use"])

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_small_fill.png", img_as_ubyte(img_small))
    s["img_mask_small_removed"] = (img_small * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = img_reduced

    s.addToPrintList("percent_small_tissue_filled",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After MorphologyModule.fillSmallHoles: NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After MorphologyModule.fillSmallHoles: NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")
    return




def compareObjects(s, params):
    # print("Comparing objects...")
    hu_thresh = 0.015
    logging.info(f"{s['filename']} - \tcompareObjects")
    area_mask = s["img_mask_use"]
    area_label = measure.label(area_mask, background = 0)
    area_props = measure.regionprops(area_label)
    cluster_labels, cluster_center, score = _get_convex_hull(area_label, area_props)

    img = s.getImgThumb(s["image_work_size"])

    obj_mask = _get_group_mask(area_props, img, cluster_labels)
    obj_label = measure.label(obj_mask, background = 0)
    obj_props = measure.regionprops(obj_label)
    s.addToPrintList("Objects", len(obj_props))


    if len(obj_props) > 1:

        # print(obj_props)
        for o_ref, obj_ref in enumerate(obj_props):
            for o_comp, obj_comp in enumerate(obj_props):
                hu_diff_hull = abs(sum(obj_ref.moments_hu - obj_comp.moments_hu))
                hu_diff_area = abs(sum(area_props[o_ref].moments_hu - area_props[o_comp].moments_hu))

                s.addToPrintList("cHull Hu diff {0} vs {1}".format(o_ref, o_comp), str(hu_diff_hull))
                s.addToPrintList("cHull log Hu diff {0} vs {1}".format(o_ref, o_comp), str(-np.log10(hu_diff_hull)))
                s.addToPrintList("mask_use Hu diff {0} vs {1}".format(o_ref, o_comp), str(hu_diff_area))
                s.addToPrintList("mask_use log Hu diff {0} vs {1}".format(o_ref, o_comp), str(-np.log10(hu_diff_area)))

        

                if hu_diff_hull < hu_thresh:
                    obj_label[obj_label == obj_comp.label] = obj_ref.label
                    obj_comp.label = obj_ref.label



    colored = color.label2rgb(obj_label, bg_label=0)

    plt.figure()
    plt.imshow(colored)
    plt.scatter(obj_props[0].centroid[1], obj_props[0].centroid[0], c="k")
    plt.scatter(obj_props[1].centroid[1], obj_props[1].centroid[0], c="k")
    plt.axis("off")
    # plt.show()

    plt.savefig(s["outdir"] + os.sep + s["filename"] + "_simCheck.png")




def _get_convex_hull(label, props):
    # get cluster information and convex hull mask for each object

    if len(props) > 3:
        centroids = np.array([p.centroid for p in props])
        cluster_labels, cluster_center, score = _find_clusters(centroids)

        label1 = label.copy()
        if score > 0.5:
            for p in range(len(props)):
                prop = props[p]
                label1[label == prop.label] = cluster_labels[p]*100
                prop.label = cluster_labels[p]

            label = label1 * 0.01

    else:
        cluster_labels = np.arange(1,len(props)+1)
        cluster_center = [p.centroid for p in props]
        score = 0

    return cluster_labels, cluster_center, score



def _find_clusters(centroids, clustering = "short"):
    
    if clustering == "short":
        range_clusters = np.arange(2, min([10, len(centroids)]))

    elif clustering == "full":   
        range_clusters = np.arange(2,len(centroids))
    
    scores = []
    clusters = []
    cluster_center = []
    for n in range_clusters:
        clusterer = KMeans(n_clusters=n, random_state=10)
        cluster_labels = clusterer.fit_predict(centroids) + 1 #add plus one to keep 0 as background "class"
        silhouette_avg = silhouette_score(centroids, cluster_labels)

        # db_score = davies_bouldin_score(centroids, cluster_labels)

        scores.append(silhouette_avg)
        clusters.append(cluster_labels)
        cluster_center.append(clusterer.cluster_centers_)
        
    best = np.argmax(scores)

    return clusters[best], cluster_center[best], scores[best]


def _get_group_mask(props, img, cluster_labels):

        g_mask = np.zeros((img.shape[0], img.shape[1]), dtype = bool)
        for c in np.unique(cluster_labels):

            # get mask for convex hull of each group of areas (each object)
            m = _calc_group_props(props, c, img)
            g_mask[m] = True

        return g_mask

def _calc_group_props(props, c, img):

    group = [p for p in props if p.label == c]

    points = np.concatenate(np.asarray([g.coords for g in group]))
    hull = ConvexHull(points)

    mask = morphology.convex_hull.grid_points_in_poly(shape=img.shape, verts=hull.points[hull.vertices])


    return mask