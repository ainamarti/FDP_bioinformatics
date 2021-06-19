#---------------------------- IMPORT ALL LIBRARIES ----------------------------#
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pylab as plt
from termcolor import colored, cprint
import seaborn as sns
import math
import random
from scipy import ndimage, misc
from PIL import Image
import os
import sys
import time
from datetime import timedelta
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.constraints import NonNeg
import multiprocessing
from tqdm.keras import TqdmCallback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Do not display tensorflow logs
