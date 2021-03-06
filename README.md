# Final Degree Project
## Aina Martí Aranda 2021
Scripts and materials used in my final graduate project: "Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline".

### USAGE
To build models with the scripts, type the following with your own variables:
```python
python3 train_model.py $copy $epochs $patience $fold $threshold $rounds $batch_size $output_name $raw $approach_folder $size
```
  * `copy` : this determines the number of copies to be generated for each original image. 0 means no data augmentation. 9 means having 10 versions of the same image (the original plus 9 copies). For example `copy=9`
  * `epochs` : Number of epochs for which the model will be trained. For example `epochs=200`
  * `patience` : number of epochs after which the process will stop training if the monitored metric has stopped improving. For example `patience=50`
  * `fold` : name of the fold to be used. For example `fold=fold1`
  * `threshold` : threshold to be used to binarize the observed values. For example `threshold=0.4`
  * `rounds` : number of rounds to be used for this model. For example `rounds=4`
  * `batch_size` : batch size to be used when training the model. For example `batch_size=30`
  * `output_name` : name of the file where the output values will be stored. For example `output_name=output.txt`
  * `raw` : if the images to be used will be in RAW format or not (DECONVOLVED). For example `raw=False`
  * `approach_folder` : name of the path where to store the files. For example  `approach_folder=approach15`
  * `size` : size of the images in pixels. For example `size=150` in this case the images will be 150x150 pixels


### FILES

  * [`/det_per_round_RAW36plex.csv`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/det_per_round_RAW36plex.csv) : This file contains all the information needed to train the models with the RAW and DECONVOLVED images. Description of the columns:
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
  * [`/execute`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/execute) : This bash file shows an example for the execution of several models. It executes the 4 models (with 4, 3, 2 and 1 rounds of sequencing) for each fold. 
  * [`/train_model.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/execute) : This is the main script. It calls the functions in the file `my_functions.py` to prepare the images, and build, train and save the models. To use the scripts, this file should be called as detailed in the [usage](https://github.com/ainamarti/FDP_bioinformatics/tree/main#usage) section.
  * [`/libraries.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/libraries.py) : This python file imports all the needed libraries.
  * [`/my_functions.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/my_functions.py) : This python file contains all the functions needed for the main script to work.
  * [`/output.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/execute) : This python file takes the output from an approach and gives the summarized results.  
  * [`/train_list.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/train_list.py) : This python file generates the indexes for all the datasets (training, folds and external validation).
  * [`/notebooks.py`](https://github.com/ainamarti/FDP_bioinformatics/blob/main/notebooks) : This folder contains several jupyter notebooks aimed to analyze the results generated and create the plots from the project.
