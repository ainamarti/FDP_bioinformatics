# # AINA MARTÍ ARANDA
# # 2021 Final Graduate Project
# # Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline


# # FUNCTIONS FOR MODEL TRAINING AND DATA PREPARATION # #
from libraries import *

#---------------------------- LOAD THE DATA NEEDED ----------------------------#
def load_data():
    '''
    This function will load the table named "det_per_round_RAW36plex.csv"
    This table contains the information needed to prepare and process all the images
    in the datasets used: category, FOV number, X and Y postions in the FOV, observed barcodes,
    roi number, chemistry used, and path to the corresponding raw images and segmented and deconvolved images
    '''

    print(colored("\n[INFO]", "cyan"),"Loading data")
    dataframe=pd.read_table('../det_per_round_RAW36plex.csv', sep=",", index_col=0)

    to_mic=0.26675324675324674
    to_pix=3.749406739439962
    def to_pixel(x):
            return x*to_pix
    dataframe["X"] = dataframe["X"].apply(to_pixel)
    dataframe["Y"] = dataframe["Y"].apply(to_pixel)

    return dataframe

#---------------------------- FUNCTIONS TO PREPARE AND GENERATE THE IMAGES ----------------------------#

def add_image(dataframe, b, row, group):
    '''
    This function will add an already prepared image to the dataframe containing all the images and their 
    information.
    ARGUMENTS:
    - dataframe: dataframe where the image will be added
    - b: prepared image stored in a numpy array
    - row: row of the original dataframe containing the information of the cell
    - group: group where the cell has been classified (train/fold/test)
    '''

    dataframe=dataframe.append({'image_array':b, 
                                        "image":row["image"], 
                                        "roi":row["roi"], 
                                        "category":row["category"], 
                                        "area":row["area"], 
                                        "perimeter":row["perimeter"], 
                                        "chemistry":row["chemistry"], 
                                        "X":row["X"], 
                                        "Y":row["Y"], 
                                        'file':row["path_raw"], 
                                        'expected':row["expected"], 
                                        'observed_r1':row["1"], 
                                        'observed_r2':row["2"], 
                                        'observed_r3':row["3"], 
                                        'observed_r4':row["4"],
                                        "group":group}, ignore_index=True)
    return dataframe

def random_padding(a, size):
    '''
    This function will add random padding to the borders of the image. 
    With this function, the cell WON'T be located always in the top-left corner!
    ARGUMENTS:
    - a: image where padding will be added to. Stored in numpy format
    - size: size of the final image after adding padding. For example, if size=150, the final image
            will be 150x150 pixels independently of its original size.
    '''
    pad_x=random.randint(0,size-a.shape[1])
    pad_y=random.randint(0,size-a.shape[2])
    
    npad = ((0,0), (pad_x, ((size-a.shape[1])-pad_x)), (pad_y, ((size-a.shape[2])-pad_y)))
    
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)
    return b

def normalize(b, rounds, channels):
    '''
    This function will normalize the pixel intensities accross channels.
    ARGUMENTS:
    b: image to be normalized. should have the z slices dimension projected
    rounds: number of rounds to be normalized in the image
    channels: number of channels to be normalized
    '''
    for r in range(rounds):#all rounds
        for c in range(channels): #all channels but not dapi
            b[r,c] = b[r,c] / np.max(b[r,c]) #normalize
    return b

def normalize_threshold(b):
    '''
    This function will normalize the pixel intensities different from zero.
    It is intended to be used after threhsolding has been applied.
    ARGUMENTS:
    b: image to be normalized.
    '''
    b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
    return b

def random_threshold(row, b):
    '''
    This function will apply a random threshold to the pixel intensities.
    the value of the threshold will be randomly selected between a range of values.
    This range is different for each dataset due to their differences in chemistries used.
    ARGUMENTS:
    row: row from the original dataframe containing iformation about the cell. This will be used
         to know the dataset to which it belongs (36plex-5K or 36plex-1K)
    b: image to be modified.
    '''
    if row['group'] == '36plex-5K LIT':
        thr_solid=random.uniform(0.85,0.95)
        b[b<thr_solid] = 0
        rand_thr=thr_solid
    else:
        thr_jeb=random.uniform(0.82, 0.92)
        b[b<thr_jeb] = 0
        rand_thr=thr_jeb
    return b, rand_thr

def correct_size(images, size):
    '''
    This function will crop the images if the sizes are higher than the specified by the user.
    This function is intended to be used afer the rotation of the images. Since rotation does not reshape the 
    size of the images, it adds a black background, thus increasing the sizes of the images. 
    ARGUMENTS
    - images: set of images from one single cell (with all the corresponding dimensions)
    '''
    def correct_size_one(image, y, x, size):
        '''
        This function will change the size image by image.
        '''
        cropx=x;cropy=y
        if x>size and y> size:
            cropx = size ; cropy = size
        elif x>size:
            cropx = size ; cropy=y

        elif y>size:
            cropx=x; cropy = size
        
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        #print(image.shape)
        image1=image[list(range(starty,starty+cropy))]
        #print(image1.shape)
        image2=image1[:,list(range(startx,startx+cropx))]
        #print(image2.shape)
        return image2
    
    y0, x0 = images.shape[-2], images.shape[-1]   
    result=list(map(lambda i: correct_size_one(i, y0, x0, size), images)) #give all images (from all dimensions) one by one to the main function
    return np.array(result)

def rotate_image(b, size):
    '''
    This function will rotate the images using a random angle selected from a range of values.
    ARGUMENTS
    - b: image to be rotated
    - size: final size of the images. It will be used to call the "correct_size" function. This will make sure that
            no image expands beyond the specified size.
    '''
    angle=random.randint(10,350)
    def rotate_one(image, angle):
        return ndimage.rotate(image, angle, reshape=True)
    
    result=np.asarray(list(map(lambda i: rotate_one(i, angle), b)))
    result=correct_size(result, size)
    return result, angle

def modify_size(b, size):
    '''
    This function will modify the size of the images using a random percentage of modification.
    The percentage will be computed taking into account:
        - the maximum size
        - the size of the image
        - the minimum size possible
    Depending on the percentage, the image will be zoomed in or zoomed out.

    ARGUMENTS
    - b: image to be rotated
    - size: final size of the images. It will be used to call the "correct_size" function. This will make sure that
            no image expands beyond the specified size.
    '''
    y0, x0 = b.shape[-2], b.shape[-1] 
    
    if y0>=x0:
        perc=size/y0
    else:
        perc=size/x0
    
    final_perc=random.uniform(0.7,perc)
    
    def size_one(b, final_perc, y0, x0):
        img=Image.fromarray(b)
        #print(x0, y0)
        #print((round(y0*final_perc),round(x0*final_perc)))
        img=img.resize((round(x0*final_perc),round(y0*final_perc)), Image.ANTIALIAS)
        return np.asarray(img)
    
    result=np.asarray(list(map(lambda i: size_one(i, final_perc, y0, x0), b)))
    return result, final_perc
    
def flip(b):
    '''
    This function will flip 60% of the images generated with data augmentation.
    ARGUMENTS
    - b: image to be flipped
    '''
    p=random.uniform(0,1)
    flipped=False
    def flip_one(i):
        b=np.fliplr(i)
        return b
    if p>0.4:
        result=np.asarray(list(map(lambda i: flip_one(i), b)))
        flipped=True
    else:
        result=b.copy()
    return result, flipped

def prepare_image1(a, channels, row, rounds, size, raw): # IMAGE PREPROCESSING 1
    '''
    This is image preprocessing procedure 1. Steps:
        projection of Z-slices using maximum intensity
        normalization accross channels
        projection of channels using maximum intensity
        padding
        thresholding
        normalization
    
    ARGUMENTS:
    - a: original image to be prepared
    - channels: number of channels to be taken into account
    - row: row of the original dataframe with the information of the cell
    - rounds: number of rounds to be taken into account
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    '''

    #step 1: PROJECT ALL Z IMAGES
    a = np.amax(a,axis=1)
    a = a[:rounds,:channels]

    #step 2: CHANNELS NORMALIZATION
    for r in range(rounds):
        for c in range(channels): #all channels but not dapi
            a[r,c] = a[r,c] / np.max(a[r,c]) #normalize
    
    #step 3: CHANNELS PROJECTION
    a = np.amax(a,axis=1) 

    #step 4: PADDING
    npad = ((0, 0), (0, (size-a.shape[1])), (0, (size-a.shape[2])))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)


    #step 5: THRESHOLDING
    if not raw:
        if row['group'] == '36plex-5K LIT':
            b[b<(59000/(2**16-1))] = 0
        else:
            b[b<(57500/(2**16-1))] = 0
    if raw:
        b[b<0.87] = 0 #with raw images
        b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
        b[b<0.50] = 0
    
    b=b[:rounds] #take only the number of rounds specified

    #step 6: NORMALIZATION
    b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
    return a, b

def prepare_image2(a, channels, row, rounds, size, raw): # IMAGE PREPROCESSING 2
    '''
    This is image preprocessing procedure 2. Steps:
        projection of Z-slices using maximum intensity
        normalization accross channels
        projection of channels using maximum intensity
        padding
        thresholding
        projection of round images with minimum intensity
        normalization
    
    ARGUMENTS:
    - a: original image to be prepared
    - channels: number of channels to be taken into account
    - row: row of the original dataframe with the information of the cell
    - rounds: number of rounds to be taken into account
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    '''
    #step 1: PROJECT ALL Z IMAGES
    a = np.amax(a,axis=1)
    a = a[:rounds,:channels]

    #step 2: CHANNELS NORMALIZATION
    for r in range(rounds):
        for c in range(channels): #all channels but not dapi
            a[r,c] = a[r,c] / np.max(a[r,c]) #normalize
    
    #step 3: CHANNELS PROJECTION
    a = np.amax(a,axis=1) 

    #step 4: PADDING
    npad = ((0, 0), (0, (size-a.shape[1])), (0, (size-a.shape[2])))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)


    #step 5: THRESHOLDING
    if not raw:
        if row['group'] == '36plex-5K LIT':
            b[b<0.928] = 0
        else:
            b[b<0.906] = 0
    
    if raw:
        b[b<0.87] = 0 #with raw images
        b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
        b[b<0.50] = 0
        

    #step 6: ROUNDS PROJECTION 
    b = np.amin(b,axis=0) 
    
    #step 7: NORMALIZATION
    if len(b[b!=0])>0:
        minimum=np.min(b[b!=0])
    else:
        minimum=0
    range_ = np.max(b)-minimum
    b[b!=0]=(b[b!=0]-minimum)/(range_)#normalize again
    range2 = 1 - 0.95
    b = (b * range2) + 0.95

    return a, b

def prepare_image3(a, channels, row, rounds, size, raw): # IMAGE PREPROCESSING 3
    '''
    This is image preprocessing procedure 3. Steps:
        projection of Z-slices using maximum intensity
        normalization accross channels
        padding
        thresholding
        normalization
    
    ARGUMENTS:
    - a: original image to be prepared
    - channels: number of channels to be taken into account
    - row: row of the original dataframe with the information of the cell
    - rounds: number of rounds to be taken into account
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    '''
    #step 1: PROJECT ALL Z IMAGES
    a = np.amax(a,axis=1)
    a = a[:rounds,:channels]
    
    #step 2: CHANNELS NORMALIZATION
    for r in range(rounds):
        for c in range(channels): #all channels but not dapi
            a[r,c] = a[r,c] / np.max(a[r,c]) #normalize


    #step 3: PADDING
    npad = ((0, 0), (0, 0), (0, (size-a.shape[2])), (0, (size-a.shape[3])))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)


    #step 4: THRESHOLDING
    if not raw:
        if row['group'] == '36plex-5K LIT':
            b[b<(59000/(2**16-1))] = 0
        else:
            b[b<(57500/(2**16-1))] = 0
    if raw:
        b[b<0.87] = 0 #with raw images
        b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
        b[b<0.50] = 0

    b=b[:rounds]

    #step 5: NORMALIZATION
    b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
    return a, b

def make_copy1(a, row, size, rounds, raw): # DATA AUGMENTATION 1
    '''
    This is data augmentation procedure 1. Steps:
        flip 60% of the copies
        rotation
        zoom in or out
        padding NOT random
        thresholding NOT random
        normalize
    
    ARGUMENTS:
    - a: image to be prepared and ready to apply modifications
    - row: row of the original dataframe with the information of the cell
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    - rounds: number of rounds to be taken into account
    - raw: If the images used are in raw form (boolean)
    '''

    #Step 4: flipping
    b, flipped=flip(a)
    
    #Step 2: rotation
    b, angle=rotate_image(b, size)

    #Step 3: size modification
    b, final_perc=modify_size(b, size)

    #Step 4: padding NOT random
    npad = ((0, 0), (0, (size-b.shape[1])), (0, (size-b.shape[2])))
    b = np.pad(b, pad_width=npad, mode='constant', constant_values=0)

    #Step 5: Thresholding NOT random
    if not raw:
        if row['group'] == '36plex-5K LIT':
            b[b<(59000/(2**16-1))] = 0
        else:
            b[b<(57500/(2**16-1))] = 0
    if raw:
        b[b<0.87] = 0 #with raw images
        b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
        b[b<0.50] = 0
    b=b[:rounds]

    #Step 6: Normalization
    for c in range(b.shape[0]):
        b[c][b[c]!=0]=(b[c][b[c]!=0]-np.min(b[c][b[c]!=0]))/(np.max(b[c])-np.min(b[c][b[c]!=0]))

    return b

def make_copy2(a, row, size, rounds, raw): # DATA AUGMENTATION 2
    '''
    This is data augmentation procedure 2. Steps:
        flip 60% of the copies
        rotation
        zoom in or out (modify size)
        padding NOT random
        Random thresholding
        normalize
        Modify (blurring, denoise, or sharpen)
    
    ARGUMENTS:
    - a: image to be prepared and ready to apply modifications
    - row: row of the original dataframe with the information of the cell
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    - rounds: number of rounds to be taken into account
    - raw: If the images used are in raw form (boolean)
    '''
    p=random.uniform(0,1)

    #Step 1: flip
    b, flipped=flip(a)
    
    #Step 2: rotate
    b, angle=rotate_image(b, size)

    #Step 3: modify size
    b, percentage_size=modify_size(b, size)

    #Step 4: padding NOT random
    npad = ((0, 0), (0, (size-b.shape[1])), (0, (size-b.shape[2])))
    b = np.pad(b, pad_width=npad, mode='constant', constant_values=0)


    #Step 5: Random thresholding
    b,rand_thr=random_threshold(row, b)
    b=b[:rounds]

    #step 6: Normalization
    for r in range(rounds):
        b[r][b[r]!=0]=(b[r][b[r]!=0]-np.min(b[r][b[r]!=0]))/(np.max(b[r])-np.min(b[r][b[r]!=0]))

    #Steo 7: Modifications
    modification="original"
    p=random.uniform(0,1)
    if p<0.3:
        b = ndimage.gaussian_filter(b, sigma=0.8)
        modification="blurred"
    elif p<0.6:
        b=ndimage.median_filter(b,2)
        modification="denoised"
    elif p<0.9:
        alpha = 10
        gauss_denoised=ndimage.gaussian_filter(b,2)
        sharpened=b.copy()
        sharpened=sharpened+alpha*(sharpened-gauss_denoised)
        modification="sharpened"
        for channel in range(sharpened.shape[0]):
            sharpened[channel]=(sharpened[channel]-np.min(sharpened[channel]))/(np.max(sharpened[channel])-np.min(sharpened[channel]))
        sharpened[b==0]=0
        b=sharpened.copy()
    
    return b

def make_copy3(a, row, size, rounds, raw): # DATA AUGMENTATION 3
    '''
    This is data augmentation procedure 3. Steps:
        flip 60% of the copies
        rotation
        zoom in or out (modify size)
        padding NOT random
        thresholding NOT random
        normalize
        Modify (blurring, denoise, or sharpen)
    
    ARGUMENTS:
    - a: image to be prepared and ready to apply modifications
    - row: row of the original dataframe with the information of the cell
    - size: final size of the images. For example, if 150, the resulting image will be 150x150 pixels
    - rounds: number of rounds to be taken into account
    - raw: If the images used are in raw form (boolean)
    '''

    #Step 1: flip
    b, flipped=flip(a)
    
    #Step 2: rotate
    b, angle=rotate_image(b, size)

    #Step 3: modify size
    b, percentage_size=modify_size(b, size)

    #Step 4: padding NOT random
    npad = ((0, 0), (0, (size-b.shape[1])), (0, (size-b.shape[2])))
    b = np.pad(b, pad_width=npad, mode='constant', constant_values=0)

    #Step 5: thresholding NOT random
    if not raw:
        if row['group'] == '36plex-5K LIT':
            b[b<(59000/(2**16-1))] = 0
        else:
            b[b<(57500/(2**16-1))] = 0
    if raw:
        b[b<0.87] = 0 #with raw images
        b[b!=0]=(b[b!=0]-np.min(b[b!=0]))/(np.max(b)-np.min(b[b!=0]))
        b[b<0.50] = 0
    b=b[:rounds]

    #Step 6: normalization
    for r in range(rounds):
        b[r][b[r]!=0]=(b[r][b[r]!=0]-np.min(b[r][b[r]!=0]))/(np.max(b[r])-np.min(b[r][b[r]!=0]))
    #ax7.imshow(b[1], cmap="gray")

    #Step 7: modifications (blurring, denoising or sharpening)
    modification="original"
    p=random.uniform(0,1)
    if p<0.3:
        b = ndimage.gaussian_filter(b, sigma=0.8)
        modification="blurred"
    elif p<0.6:
        b=ndimage.median_filter(b,2)
        modification="denoised"
    elif p<0.9:
        alpha = 10
        gauss_denoised=ndimage.gaussian_filter(b,2)
        sharpened=b.copy()
        sharpened=sharpened+alpha*(sharpened-gauss_denoised)
        modification="sharpened"
        for channel in range(sharpened.shape[0]):
            sharpened[channel]=(sharpened[channel]-np.min(sharpened[channel]))/(np.max(sharpened[channel])-np.min(sharpened[channel]))
        sharpened[b==0]=0
        b=sharpened.copy()
    
    return b

def generate_images(dataframe, copies, rounds,channels,  images_train, images_fold, train_index, fold_used, size, total, raw):
    '''
    This function prepares the images for the models. It generates the originals and makes the corresponding copies
    
    ARGUMENTS:
     - dataframe: pandas dataframe with all the images information (path, group, roi, category, etc)
     - copies: number of copies to generate for every original image
     - rounds: number of rounds of sequencing to be taken
     - channels: number of channels to be considered (usually 4)
     - images_train: pandas dataframe where the training images will be stored with their corresponding information.
     - images_fold: pandas dataframe where the internal validation images will be stored with their corresponding information.
     - train_index: index of images that will be used for the training set.
     - fold_used: list of image indexes that belong to the internal validation in the CV fold used.
     - size: final size of the images (if 150, the final images will be 150x150 pixels)
     - total: total number of images in the dataframe (1171)
     - raw: if the images will be used in raw format or already deconvolved.
        
    '''
    
    for i, row in dataframe.iterrows():

        group=""
        #define to which group each image belongs
        if i in train_index:
            if i in fold_used:
                group="fold"
            else:
                group="train"
        else:
            group="test"
        
        # prepare the images from the training and internal validation datasets.
        if group!="test":
            if raw: # if we want to use the raw format, take the path to the raw image
                a = tiff.imread("../raw_images/"+row['path_raw'])
            else: # otherwise take the path to the deconvolved images
                a = tiff.imread(row['path_decon'])
            
            a = a.astype(np.float32)

            # prepare the images with the corresponding function.
            # prepare_image1 (IP1), prepare_image_2 (IP2) or prepare_image3 (IP3)
            a,b=prepare_image1(a, channels, row, rounds, size, raw)

            if group=="fold":
                images_fold=add_image(images_fold, b, row, group)
            if group=="train":
                images_train=add_image(images_train, b, row, group)

                #DATA AUGMENTATION: generate new images from the previous one
                #This is done only for the training set!!!
                for copy in range(copies):
                    # Make the copies of the images with the corresponding function.
                    # make_copy1 (IP1), make_copy2 (IP2) or make_copy3 (IP3)
                    b=make_copy1(a, row, size, rounds, raw)
                    if group=="fold": 
                        images_fold=add_image(images_fold, b, row, group)
                    if group=="train":
                        images_train=add_image(images_train, b, row, group)
                
    return images_train, images_fold



#---------------------------- FUNCTIONS TO TRAIN THE MODEL ----------------------------#

#------------ MODEL ARCHITECTURES -----------#

def cnn2D_1(width, height, depth, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 1
    '''
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(10, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg(), activation='sigmoid'))
    
    return model

def cnn2D_2(width, height, depth, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 2
    '''
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg()))
    
    return model

def cnn2D_3(width, height, depth, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 3
    '''
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(126, (3, 3), activation='relu'))
    model.add(layers.Conv2D(126, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg(), activation='sigmoid'))
    
    return model

def cnn2D_4(width, height, depth, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 4
    '''
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(126, (3, 3), activation='relu'))
    model.add(layers.Conv2D(126, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(258, (3, 3), activation='relu'))
    model.add(layers.Conv2D(258, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg(), activation='sigmoid'))
    
    return model

def cnn3D_3(width, height, channels, rounds, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 3 in 3 dimensions
    '''
    inputShape = (height, width, channels, rounds)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv3D(32, (3, 3, 1), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Conv3D(64, (3, 3, 1), activation='relu'))
    model.add(layers.Conv3D(64, (3, 3, 1), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Conv3D(126, (3, 3, 1), activation='relu'))
    model.add(layers.Conv3D(126, (3, 3, 1), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg(), activation='sigmoid'))
    
    return model

def cnn3D_4(width, height, depth, rounds, final_dense=4, concat_model=False):
    '''
    this function initializes model architecture 4 in 3 dimensions
    '''
    inputShape = (height, width, depth, rounds)
    chanDim = -1
    # define the model input
    model = models.Sequential()
    model.add(layers.Conv3D(32, (3, 3, 1), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Conv3D(64, (3, 3, 1), activation='relu'))
    model.add(layers.Conv3D(64, (3, 3, 1), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Conv3D(126, (3, 3, 1), activation='relu'))
    model.add(layers.Conv3D(126, (3, 3, 1), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 1)))
    model.add(layers.Conv3D(258, (3, 3, 1), activation='relu'))
    model.add(layers.Conv3D(258, (3, 3, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    if not concat_model:
        model.add(layers.Dense(final_dense, kernel_constraint=NonNeg(), activation='sigmoid'))
    
    return model

#------------ PREPARE SUBSETS  -----------#
def prepare_dataset_1(dataframe, use_round):
    '''
    This function makes the last preparations for the images to be used by the model.
    It changes the axis of the images and computes the percentages for the labels.
    This function will be used when prepare_image1 is used
    '''
    x = []; y = []
    for index, row in dataframe.iterrows():
        img=np.moveaxis(row["image_array"],0, 2)
        x.append(img)
        y.append(row['observed_r'+str(use_round)]/row['expected'])#percentage of detection/100
        #print(row['observed_r'+str(use_round)], row['expected'])
    #print(y)
    x = np.asarray(x)
    y = np.asarray(y)

    return x, y
    
def prepare_dataset_2(dataframe, use_round):
    '''
    This function makes the last preparations for the images to be used by the model.
    It changes the axis of the images and computes the percentages for the labels.
    This function will be used when prepare_image2 is used
    '''
    x = []; y = []
    for index, row in dataframe.iterrows():
        img=np.expand_dims(row["image_array"], axis=2)
        x.append(img)
        y.append(row['observed_r'+str(use_round)]/row['expected'])#percentage of detection/100
        #print(row['observed_r'+str(use_round)], row['expected'])
    #print(y)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

def prepare_dataset_3(dataframe, use_round):
    '''
    This function makes the last preparations for the images to be used by the model.
    It changes the axis of the images and computes the percentages for the labels.
    This function will be used when prepare_image3 is used
    '''
    x = []; y = []
    for index, row in dataframe.iterrows():
        img=np.moveaxis(row["image_array"],0, 3)
        img=np.moveaxis(img, 0, 3)
        x.append(img)
        y.append(row['observed_r'+str(use_round)]/row['expected'])#percentage of detection/100
        #print(row['observed_r'+str(use_round)], row['expected'])

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


#------------ BUILD AND EVALUATE THE MODELS  -----------#
def model(use_round, train, val, batch_size, patience, size, EPOCHS):
    '''
    This function will call the functions to prepare the images and the datasets
    It generates the ouptut to follow the progress and status of the computations
    ARGUMENTS:
     - use_round: number of rounds to be used
     - train: prepared training dataset with the images and their corresponding information (prepared with function prepare_images1 for example)
     - val: prepared validation dataset with the images and their corresponding information
     - batch_size: batch size to be used to fit the model
     - patience: number of epochs to wait until the validation set stops improving
     - size: size of the images (if 150, the images are supposed to be 150x150 pixels)
     - EPOCHS: maximum number of epochs to train the model
    '''

    cprint('\n#### MODEL USING '+ str(use_round)+' ROUNDS ####',"grey",'on_cyan', attrs=['reverse'])
    
    ### SPLIT TEST AND TRAIN ###
    x_train, y_train = prepare_dataset_1(train, use_round)
    x_val, y_val = prepare_dataset_1(val, use_round)
    print("number of samples in train:", len(y_train))
    print("number of samples in validation:", len(y_val))
    

    ### BUILDING AND FITTING THE MODEL ###
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    depth=use_round
    #depth=1  ## This will be used if we project the rounds
    ch=4 

    model = cnn2D_1(size, size, depth, final_dense=1) # change this function depending on the model architecture to be used
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=['mean_squared_error'])
    print()
    print(colored("[INFO]", "cyan"), "fitting the model...")

    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stop],
                        batch_size=batch_size)

    return  model, history, x_val, y_val

def evaluate_model(x_test, y_test, model, threshold):
    '''
    This function will evaluate the model on an additional dataset.
    It computes the mean squared error, the values needed to plot the ROC curve, 
    the predicted values and the AUC
    ARGUMENTS:
     - x_test: imeges from the dataset to be evaluated
     - y_test: labels from the dataset to be evaluated
     - model: model previously trained
     - threshold: threshold to be used with the observed values (labels)
    '''

    loss = model.evaluate(x_test, y_test)
    print(colored("[METRICS]", "blue"), "Loss: ", loss)
    preds = model.predict(x_test)
    testy=y_test>threshold
    lr_auc = roc_auc_score(testy, preds)
    print(colored("[METRICS]", "blue"), 'ROC AUC=%.3f' % (lr_auc))
    lr_fpr, lr_tpr, thresholds = roc_curve(testy, preds)

    return lr_auc, lr_fpr, lr_tpr, thresholds, loss, preds

def create_plots( approach, lr_fpr, lr_tpr, thresholds, history, y_test, preds, use_round, plot_name_ROC, result_model_name):
    '''
    this function will create and save three different plots:
        * the ROC curve for classification
        * Observed vs. Predicted percentage detection
        * LOSS value history through epochs
    ARGUMENTS:
     - approach: name of the approach that is being tested
     - lr_fpr: list of false positive rates to build the ROC curve
     - lr_tpr: list of true positive rates to build the ROC curve
     - thresholds: list of thresholds used (to build the ROC curve)
     - history: history of LOSS values through epochs
     - y_test: observed labels
     - preds: predicted labels
     - use_round: number of rounds used
     - plot_name_ROC: name for the file saving the ROC curve
     - result_model_name: name for the file saving the loss history and obs vs pred plots
    '''
    pyplot.figure(figsize=(13, 10))
    pyplot.axis('equal')
    lims = [0, 1]
    pyplot.plot(lr_fpr, lr_tpr, c="black", lw=2, label='ROC curve')
    pyplot.scatter(lr_fpr, lr_tpr, c=thresholds, cmap='brg')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="random chances")
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlim(lims)
    pyplot.ylim(lims)
    # show the legend
    pyplot.legend()
    pyplot.colorbar()
    # show the plot
    pyplot.title("ROC curve for classification")
    pyplot.savefig(approach+plot_name_ROC)

    pyplot.show()

    print("Plot of the ROC curve generated")

    #plot results
    print(colored("[INFO]", "cyan"), "plotting results ...", end=" ")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(25)

    fig.suptitle('Number of rounds: '+str(use_round), fontsize=25)
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('model loss', fontsize=22)
    ax1.set_xlabel('epoch',fontsize=20)
    ax1.set_ylabel('loss',fontsize=20)
    ax1.set_ylim([0,0.2])
    ax1.tick_params(labelsize=18)
    ax1.legend(['train', 'test'], loc='upper left')
    
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    ax2.set_aspect('equal')
    sns.scatterplot(x=np.array(y_test), y=np.array(preds)[:,0],
                    palette=cmap
                   )


    ax2.set_title("Observed vs. Predicted perc detection",
                    fontsize=22, pad=20)

    ax2.set_ylabel("Predicted %",
                     fontsize=20)

    ax2.set_xlabel('Observed %',
                     fontsize=20)

    ax2.tick_params(labelsize=18)

    lims = [0, 1.1]
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    _=ax2.plot(lims,lims, '--', color="0.8", dashes=(5, 5))
    plt.savefig(approach+result_model_name)
    print("plot saved")