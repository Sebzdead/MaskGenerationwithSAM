# SAM Trainer

The SAM trainer is a tool based of Meta's Segment Anything Model (SAM) for purpose of fine tuning/training the model for a specific task or manually extracting masks for other purposes. 

Follow the installation instructions from the [SAM repository](https://github.com/facebookresearch/segment-anything/tree/main) and download the vit_h checkpoint.

## Generating Masks for Training
Open the GenerateTrainMasks.ipynb notebook and follow the instructions to generate masks for training. This tool uses a rectangle to select the area of interest and then let's the model create the respective masks.

## Measure_masks for extracting pixel intensity values