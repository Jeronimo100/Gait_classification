# AbnormalGAIT

## HOW TO RUN

Prerequisites: python3, numpy, sklearn, pickle and enough memory to run a few tests.
The file to be run is tests.py and does not take any arguments but needs the skeleton data in the right directory (see section Data Downloads).
Running tests.py will generate all results that we reported in the report and write it to two separate readable files Stats_results.txt and Stats_results_partial.txt for the two kind of data preprocessing we performed.

## Data Downloads

http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/

Scroll to bottom of page and download <skeletons.zip> 

Move <DIRO_skeletons.npz> to /resources

## Comments

We assume your read the report to understand fully grasp following comments (only PDF file in AbnormalGAIT/ ).

The code is split over three files.

extractFeaturesSkeletonData.py contains all functions that partition and preprocess the orginial data by calculating and create the labeled feature vectors. It also contains the function called during testing that will return the aggregated prediction on sequences.

trainModels.py contains the creation and training of the different models for given training and testing data. It saves computed models to avoid recomputation and tries to load existing saved models if available (majority are alreaady available on github).

tests.py calls the testing of all classifiers for all mentioned sliding window sizes and writes the results out to the text files.
