import os
import sys
import cv2
import pickle
import numpy as np
import pathlib as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

rect_points = []


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


class trainingMaskGenerator():
    """
    This class is used to create masks that can be used for measurement of pixel intensity or as training inputs for further SAM training.
    It is assumed that the SAM model is in the same directory as the script that is calling this class.
    """

    def __init__(self) -> None:
        sys.path.append("..")
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def generate_mask_from_area(self, image_folder_path: str, mask_folder_path: str, image_resize_factor: int = 1):
        """This function is used to generate masks for the purposes of training a SAM model or for pixel intensity measurement.
        
        Parameters
        ----------
        image_folder_path: str 
            The path to the folder containing the images from which masks are to be generated.
        mask_folder_path: str
            The path to the folder where the masks are to be saved.
        image_resize_factor: int
            The factor by which the image is to be resized (i.e. image size / factor). Default is 1.
        """
        mask_folder = pl.Path(mask_folder_path)
        image_file_folder = pl.Path(image_folder_path)
        image_files = os.listdir(image_file_folder)
        # Function to store and draw rectangles based on mouse clicks
        temp_img = np.zeros((100, 100, 3), dtype=np.uint8)

        def draw_rectangle(event, x, y, flags, param):
            global rect_points, input_box

            if event == cv2.EVENT_LBUTTONDOWN:
                rect_points = [(x, y)]  # Store starting point

            elif event == cv2.EVENT_LBUTTONUP:
                rect_points.append((x, y))  # Store ending point

                # Draw rectangle on the image
                cv2.rectangle(
                    temp_img, rect_points[0], rect_points[1], (0, 255, 0), 2)
                cv2.imshow("Image", temp_img)

                # Update input_box with the coordinates
                input_box = np.array([rect_points[0][0] * image_resize_factor, rect_points[0][1] * image_resize_factor,
                                     rect_points[1][0] * image_resize_factor, rect_points[1][1] * image_resize_factor])

        # completed = 2
        for img in tqdm(image_files):
            image = cv2.imread(image_file_folder / img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            temp_img = image.copy()
            new_width = int(temp_img.shape[1] / image_resize_factor)
            new_height = int(temp_img.shape[0] / image_resize_factor)
            temp_img = cv2.resize(temp_img, (new_width, new_height))
            cv2.namedWindow("Image")
            cv2.setMouseCallback("Image", draw_rectangle)
            cv2.imshow("Image", temp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks[0], plt.gca())
            show_box(input_box, plt.gca())
            plt.axis('off')
            plt.show()
            # save the masks[0] as a pickle file
            with open(mask_folder / f"{img}.pkl", 'wb') as f:
                pickle.dump(masks[0], f)