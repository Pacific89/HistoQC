import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os



def _reset_figure():
        fig = plt.figure()
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        ax_list = [ax1, ax2]

        return fig, ax_list



def _cut_bbox(obj, img):

    array_ = img[obj.bbox[0]:obj.bbox[2],obj.bbox[1]:obj.bbox[3]]
    array = np.array(Image.fromarray(np.array(array_)).convert("L"))
    
    return array


def calc_sim_surf(ref_obj, comp_obj, img, s):

    # ref_img = cut_bbox(ref_obj, 432, 1024, remove_background=False)
    # comp_img = cut_bbox(comp_obj, 432, 1024, remove_background=False)

    ref_img = _cut_bbox(ref_obj, img)
    comp_img = _cut_bbox(comp_obj, img)

    fig, ax_list = _reset_figure()


    h_thresh = 400
    n_oct = 10
    n_oct_layers = 10
    # pil_patch = Image.fromarray(patch).convert("L")
    rotations = np.array([0, 45, 90, 135, 180])
    rot_scores = np.zeros(rotations.shape)
    pil_patch = Image.fromarray(comp_img)
    total_match_metric = 0.0

    current_best = 0
    for pos, rot in enumerate(rotations):

        patch = np.asarray(pil_patch.rotate(rot))
        surf = cv2.xfeatures2d_SURF.create(hessianThreshold=h_thresh, extended=True, nOctaves=20, nOctaveLayers=20)

        kp1, des1 = surf.detectAndCompute(ref_img, None)
        kp2, des2 = surf.detectAndCompute(patch, None)

        # Brute Force Matching ALL vs ALL

        bf = cv2.BFMatcher()

        if type(des1) == type(None) or type(des2) == type(None):
            return 0

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > 3: 
            # print("Good > 0")
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,20.0)
            matchesMask = mask.ravel().tolist()
            homography_matches = sum(matchesMask)
            rot_scores[pos] = homography_matches

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask[:min(len(matchesMask), 150)], # draw only inliers
                        flags = 2)

            if rotations[pos] == 0:
                img3 = cv2.drawMatches(ref_img,kp1,patch,kp2,good[:min(len(matchesMask), 150)],None, **draw_params)
                ax_list[1].imshow(img3)

            # plt.show()

            if rot_scores[pos] > current_best:
                # plt.title("N_Octaves: {0} \n N_oct_layers: {1} \n Total Matches: {2}".format(octaves, octaves, matches_metric))
                current_best = rot_scores[pos]

            # penalty for each angle with zero matches
            if homography_matches == 0:
                total_match_metric -= 1/len(rotations)

            # implement metric between 0 and 1 (from master thesis: deduplication)
            total_match_metric += (1/len(rotations))*min(1.0, homography_matches/min(len(des1), len(des2), 200))


            # Plot if report is wanted
            if ax_list[0] != None:

                best_rotation_pos = np.argmax(rot_scores)
                stats = [[int(current_best)], [rotations[best_rotation_pos]], [sum(rot_scores != 0)]]
                print(rot_scores)

                ax_list[1].set_title(" \n Rotation: {0} ° \n Homography Matches: {1}".format(rotations[0], rot_scores[0]))
                stat_labels = ["Best score", "Angle", "Angles != 0"]
                ax_list[0].table(cellText = stats, rowLabels = stat_labels, loc="center", cellLoc = "left", colWidths = [0.3])
                # ax[0].axis("tight")
                ax_list[0].axis("off")
                ax_list[0].title.set_text("OpenCV Scores/Matching")
    

    plt.savefig(s["outdir"] + os.sep + s["filename"] + "_simCheck_surf.png")
    return total_match_metric