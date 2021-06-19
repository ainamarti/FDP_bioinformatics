# Final Degree Project
## Aina Martí Aranda 2021
Scripts and materials used in my final graduate project: "Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline".


### /det_per_round_RAW36plex.csv:

This file contains all the information needed to train the models with the RAW images. Description of the columns:
	* X: X position in the image in microns (centroid)
	* Y: Y position in the image in microns (centroid)
	* Area: total area of the roi in the raw image
	* Perim.: size of the perimeter of the cell
	* roi: Roi number in the raw image
	* Image: image number in the replicate.
	* category: biological replicate plus technical replicate
	* Brep: biological replicate.
	* Trep: theoretical replicate.
	* Group: chemistry group or dataset. Either 36plex-1K eLIT or 36plex-5K LIT
	* Chemistry: chemistry used in this dataset. Either JEB or SOLiD
	* path_decon: path to the deconvolve image of the cell
	* path_raw: path to the raw image (i.e. without deconvolution) of the cell. These images are all in the /raw_images/merged/ folder.
	* expected: number of expected barcodes to be detected.
	* 1: number of barcodes detected only with the first round
	* 2: number of barcodes detected with the first and second rounds.
	* 3: number of barcodes detected with the 1st, 2nd, and 3rd rounds.
	* 4: number of barcodes detected with the 1st, 2nd, 3rd, and 4th rounds.
