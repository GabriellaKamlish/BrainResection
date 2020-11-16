# Weekly Meeting Notes

## Meeting 13/11
### Discussion:
- Discussed Highresnet paper, and resolved queries
- Suggested 3 solutions for the problem:
1. (Original) Design a model with many layers to be able to bypass the need for additional processing features or segmentations.
2. Attempt to use inpainting (such as by Nvidia or Deoldify) to refill the resected area. Then perform segmentation on this new image, to identify the labels and percentages lost by the resection.
3. Use the context map technique from the "Seeing what's not there" paper, and identify where there each parcel should be and whether or not it is there (Unsure how this will work with percentages and with multiple parcels)


### Tasks:
- Use highresnet and UCL GIF algorithm to parcellate both t1, and t1_resected images in order to visualise current issues with the segmentation process
- Begin PyTorch tutorial 
- Use the dataset provided on the hemisphere identifier I created
