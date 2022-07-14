import os
from io import BytesIO
from absl import flags

import src.config
import sys
import tarfile
import tempfile
from six.moves import urllib


import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2
import pdb
import glob
import argparse
from demo import main
import tensorflow as tf


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        #"""Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def StartProgress(progress_var):
    # start progress
    progress_var.start(10)


def StopProgress(progress_var):
    # stop progress
    progress_var.stop()


def makeform(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field+": ", anchor='w')
        ent = Entry(row)
        ent.insert(0, "0")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
    return entries


def select_image(entries):
    # grab a reference to the image panels
    global panelA, panelB, progress_var, L
    row = Frame(root)
    L = Label(row, width=22, text="In Process...", anchor='w')
    progress_var = ttk.Progressbar(
        row, orient=HORIZONTAL, length=300, maximum=100, mode='determinate')
    row.pack(side=TOP, fill=X, padx=5, pady=5)
    L.pack(side=LEFT)
    progress_var.pack(side=RIGHT)
    StartProgress(progress_var)
    # open a file chooser dialog and allow the user to select an input
    height = float(entries['Height In cm'].get())
    name = entries['Name'].get()
    print(name)
    print(height)
    # image
    path = tkinter.filedialog.askopenfilename()
    print(path)
    # MODEl STARTED

    if len(path) > 0:

        dir_name = path

        ## setup ####################

        LABEL_NAMES = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ])

        FULL_LABEL_MAP = np.arange(
            len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

        # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
        MODEL_NAME = 'xception_coco_voctrainval'

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
                'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
                'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

        model_dir = 'deeplab_model'
        if not os.path.exists(model_dir):
            tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, _TARBALL_NAME)
        if not os.path.exists(download_path):
            print('downloading model to %s, this might take a while...' %
                  download_path)
            urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                       download_path)
            print('download completed! loading DeepLab model...')

        MODEL = DeepLabModel(download_path)
        print('model loaded successfully!')

        #######################################################################################

        #list_im=glob.glob(dir_name + '/*_img.png'); list_im.sort()

        # for i in range(0,len(list_im)):

        image = Image.open(dir_name)
        #print("Image Type = ",type(image))
        back = cv2.imread(
            'sample_data/input/background.jpeg', cv2.IMREAD_COLOR)

        res_im, seg = MODEL.run(image)

        seg = cv2.resize(seg.astype(np.uint8), image.size)
        mask_sel = (seg == 15).astype(np.float32)
        mask = 255*mask_sel.astype(np.uint8)

        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        res = cv2.bitwise_and(img, img, mask=mask)
        bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        image = img
        edged = bg_removed
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        edged = cv2.resize(edged, (0, 0), fx=0.5, fy=0.5)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            progress_var.destroy()
            L.destroy()
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            progress_var.destroy()
            L.destroy()
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged

    # cv2.imshow("original image",img)
    # cv2.imshow("mask",res)
    # cv2.imshow('input image',bg_removed)
    # cv2.waitKey(0)

    #print("after processing = ",type(np.asarray(255*mask_sel)))
    #
    #
    #print("back type = ",type(back))
    #print("image type = ",type(np.asarray(image)))
    #print("masksDL type = ",type(255*mask_sel.astype(np.uint8)))
    #
    #
    #print("back shape = ", back.shape)
    #print("image shape = ",np.asarray(image).shape)
    #print("masksDL shape = ",255*mask_sel.astype(np.uint8).shape)

    #back_align = alignImages(back, np.asarray(image), cv2.cvtColor(255*mask_sel.astype(np.uint8),cv2.COLOR_GRAY2RGB))

    #bg_removed = remove_bg(np.asarray(image), back_align,cv2.cvtColor(255*mask_sel.astype(np.uint8),cv2.COLOR_GRAY2RGB))

    #config = flags.FLAGS
    # config(sys.argv)

    # Using pre-trained model, change this to use your own.
    #config.load_path = src.config.PRETRAINED_MODEL
    #
    #config.batch_size = 1

    # cv2.imwrite(dir_name.replace('img','back'),remove_bg)

    ######################################################
    # main(bg_removed,height,None)
    ######################################################

    #name= dir_name.replace('img','masksDL')
    # cv2.imwrite(name,(255*mask_sel).astype(np.uint8))
    # cv2.imwrite(dir_name.replace('img','back'),back_align)

    #str_msg='\nDone: ' + dir_name
    # print(str_msg)
    # MODEL ENDED

    # ensure a file path was selected


if __name__ == '__main__':
    fields = ('Name', 'Height In cm')
    # initialize the window toolkit along with the two image panels
    root = Tk()
    ents = makeform(root, fields)
    L = None
    progress_var = None
    panelA = None
    panelB = None
    print(ents)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))

    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    btn = Button(root, text="Select an image",
                 command=(lambda e=ents: select_image(e)))
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    # kick off the GUI
    root.mainloop()
