"""
Mask R-CNN
Train on the FOD-dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 fod.py train --dataset=FOD-dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 fod.py train --dataset=FOD-dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 fod.py train --dataset=FOD-dataset --weights=imagenet

    # Apply color splash to an image
    python3 fod.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 fod.py splash --weights=last --video=<URL or path to file>
"""
    
import os
import sys
import json
import numpy as np
import skimage.draw


ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("FOD/FOD-dataset"):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.model as modellib
import mrcnn.utils as utils


COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class FODConfig(Config):
    NAME = "fod"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 9
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class FODDataset(utils.Dataset):
    def load_fod(self, dataset_dir, subset):
        self.add_class("fod", 1, "flange")
        self.add_class("fod", 2, "bagstrap")
        self.add_class("fod", 3, "wrench")
        self.add_class("fod", 4, "bolt and nut")
        self.add_class("fod", 5, "led flange")
        self.add_class("fod", 6, "pipe")
        self.add_class("fod", 7, "screwdriver")
        self.add_class("fod", 8, "spanner")
        self.add_class("fod", 9, "anomaly")
        
        self.class_names = ["bg", "flange", "bagstrap", "wrench", "bolt and nut", "led flange", "pipe", "screwdriver", "spanner", "anomaly"]
        
        # "Outdoor-pipe-40__rotated_2.jpg125013":
        #     {"filename":"Outdoor-pipe-40__rotated_2.jpg",
        #      "size":125013,
        #      "regions":[{"shape_attributes":{"name":"rect","x":250,"y":329,"width":8,"height":15},
        #                  "region_attributes":{"name":"anomaly","type":"Foreign object"}}],"file_attributes":{"caption":"","public_domain":"no","image_url":""}}
        
        
        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        annotations = json.load(open(os.path.join(dataset_dir, "FOD.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                
        
            class_name = a['regions'][0]['region_attributes']['name']
            
            class_id = self.class_names.index(class_name)
            
            image_id = a['filename']
           

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "fod", 
                image_id=image_id, 
                path=image_path, 
                width=width, 
                height=height, 
                polygons=polygons,
                class_id = class_id
                )

            
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "fod":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        class_ids = np.array([info['class_id'] for _ in info['polygons']], dtype=np.int32)
        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.rectangle((p['y'], p['x']), extent=(p['height'], p['width']))
            mask[rr, cc, i] = 1
            
        return mask, class_ids
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fod":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
def train(model):
    dataset_train = FODDataset()
    dataset_train.load_fod(args.dataset, "train")
    dataset_train.prepare()
    
    dataset_val = FODDataset()
    dataset_val.load_fod(args.dataset, "val")
    dataset_val.prepare()
    
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    # Call the test function 
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect FOD.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="FOD/FOD-dataset",
                        help='Directory of the FOD-dataset')
    parser.add_argument('--weights', required=True,
                        metavar="mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
    
    #Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    # Configurations
    if args.command == "train":
        config = FODConfig()
    else:
        class InferenceConfig(FODConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weight_path = "mask_rcnn_coco.h5"
        if not os.path.exists(weight_path):
            utils.download_trained_weights(weight_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weight_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weight_path = model.get_imagenet_weights()
    else:
        weight_path = args.model

    # Load weights
    print("Loading weights ", weight_path)
    if args.weights.lower() == "coco":
        model.load_weights(weight_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weight_path, by_name=True)
        

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                               video_path=args.video)
    else: 
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))