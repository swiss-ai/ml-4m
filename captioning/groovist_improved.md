## I. Data format
This code has been tested using the VWP dataset. The original Groovist code assumes that this data is formatted `<image-sequence, story>` pairs.
Computing reverse Groovist **requires** pairs like `<[image_0, image_1, ... image_n], [statement_0, statement_1, ... statement_n]>` where the story is broken up into individual statements that correspond with the images in the sequence.

## II. Image regions
For the VWP dataset, the original Groovist code uses image regions extracted using the FasterRCNN model. These bounding boxes and labels are provided in `data/vwp_entities.csv`. I have also extracted image regions using YOLOv5, which are in `data/yolo_image_regions.csv`. This file path should be set in the configuration file (config.ini) as the `image_regions_info_file`. The path to the folder containing the cropped bounding box images should be set as the `image_regions`. 

## III. Extract noun phrases
Use the following command to extract noun phrases by statement:
`python extract_nphrases_by_statement.py --input_file data/sample_stories.json --output_file data/sample_nphrases.json`

## IV. Compute Groovist scores
The following command will save Groovist scores to the output file path.
`python groovist.py --dataset VWP --input_file data/sample_nphrases.json --output_file data/sample_scores.json`

The following Groovist scores will be saved to the output file:
- Original: original Groovist score, without concreteness weights
- Original weighted: with concreteness weights
- Reverse
- Combined: harmonic mean of normalized Original and Reverse scores
- Original, Reverse, Combined normalized: inverse tangent of the scores to normalize to [-1, 1]