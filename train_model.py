# # AINA MARTÍ ARANDA
# # 2021 Final Graduate Project
# # Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline

'''
This script will generate and save the models by calling the functions in the file "my_functions.py"
'''


print("importing libraries and functions..\n")
from libraries import *
from my_functions import *

#--------  Print the arguments given to this script ----------
print(colored("------------ARGUMENTS----------", "green"))
copies=int(sys.argv[1])
if copies>0:
    print("Using Data Augmentation with", copies, "copies")
EPOCHS=int(sys.argv[2])
print("Epochs:", EPOCHS)
patience=int(sys.argv[3])
print("Patience:", patience)
fold=sys.argv[4]
threshold=float(sys.argv[5])
print("Threshold:", threshold)
rounds=int(sys.argv[6])
batch_size=int(sys.argv[7])
print("Batch size:", batch_size)
out_file=sys.argv[8]
approach=sys.argv[10]+"/"
print("Approach: ", approach)
out_file=approach+out_file
print("Output file:", out_file)
raw=bool(int(sys.argv[9]))
if raw:
    print("Using RAW images!")
else:
    print("Using deconvolved images!")

size=int(sys.argv[11])
print("Image size: ", size)



#--------  Save the arguments to a file  ----------
with open(approach+"arguments.txt", "a") as arg: #append to the file
    arg.write('\n------ MODEL WITH '+ str(copies)+' COPIES, FOLD '+fold[-1] + ' , USING '+ str(rounds) + ' ROUNDS ----------\n')
    if copies>0:
        arg.write("Using Data Augmentation with "+str(copies)+" copies\n")
    arg.write("Epochs: "+str(EPOCHS)+"\n")
    arg.write("Patience: "+str(patience)+"\n")
    arg.write("Threshold: "+ str(threshold)+"\n")
    arg.write("Batch size: "+str(batch_size)+"\n")
    arg.write("Output file: "+out_file+"\n")
    if raw:
        arg.write("Using RAW images!")
    else:
        arg.write("Using deconvolved images!"+"\n")
    arg.write("Image size: "+str(size))


np.random.seed(37)

#--------  prepare some variables  ----------
name_model="_copies"+str(copies)+"_r"+str(rounds)+"_"+fold
channels=4
total_images=1171

#--------  write in a file the line to store the results of this model  ----------
cprint('\n------ MODEL GENERATING '+ str(copies)+' COPIES, FOLD '+fold[-1] + ' , USING '+ str(rounds) + ' ROUNDS ----------',"grey",'on_red', attrs=['blink','reverse', 'bold'])
with open(out_file, "a") as out:
        out.write("> "+str(copies)+"\t"+str(fold)+"\t"+str(rounds)+"\n")
dataframe=load_data()


#--------  prepare the dataframes where we will store the images  ----------
print(colored("[INFO]", "cyan"),"Image Preprocessing and Data augmentation")
images_train = pd.DataFrame(columns=["image_array", "image", "roi", "category", "area", "perimeter", "chemistry", "X", "Y", "file", 'expected', 'observed_r1', 'observed_r2', 'observed_r3', 'observed_r4', "group"])
images_fold = pd.DataFrame(columns=["image_array", "image", "roi", "category", "area", "perimeter", "chemistry", "X", "Y", "file", 'expected', 'observed_r1', 'observed_r2', 'observed_r3', 'observed_r4', "group"])


#--------  load the files containing the indexes to the images in each group  ----------
train_index=np.load('train_index.npy')
fold_used=np.load(fold+".npy") #this will be for example fold1.npy

#--------  generate and prepare the images  ----------
images_train, images_fold=generate_images(dataframe, copies, rounds, channels, images_train, images_fold, train_index, fold_used, size, total_images, raw)


#--------  plot the images and save them in a file  ----------
rnds=rounds
#rnds=1
print(colored("\n[INFO]", "cyan"),"Plotting train images to check...")
plt.figure(figsize=(15, 40))
for image in range(12):
    for r in range(rnds):
        plt.subplot(12,rnds,(rnds*image+r)+1)
        plt.imshow(images_train.iloc[image]["image_array"][r], cmap="gray")
plt.savefig(approach+"images_train"+name_model+".png")
plt.show()

print(colored("[INFO]", "cyan"),"Plotting validation images to check...")
plt.figure(figsize=(15, 40))
for image in range(12):
    for r in range(rnds):
        plt.subplot(12,rnds,(rnds*image+r)+1)
        plt.imshow(images_fold.iloc[image]["image_array"][r], cmap="gray")
plt.savefig(approach+"images_val"+name_model+".png")
plt.show()


#--------  output the number of images in each set  ----------
print(colored("\n[INFO]", "cyan"), "Data generated!", end=" ")
print(", Number of images: "+str(len(images_train)+len(images_fold)))
print("Number of images train:\t"+str(len(images_train)))
print("Number of images validation:\t"+str(len(images_fold)))
print("number of copies = "+str(copies))


#--------  CREATE THE MODEL  ----------

print()
print()
images_train = images_train.sample(frac = 1) #shuffle the train set
tf.keras.backend.clear_session()
model, history, x_val, y_val = model(rounds, images_train, images_fold, batch_size, patience, size, EPOCHS)

#--------  SAVE THE MODEL AND THE SUMMARY  ----------
with open(approach+'modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("model summary saved")
model.save(approach+'Model'+str(copies)+"_r"+str(rounds)+"_"+fold)


#--------  evaluate the model on the internal validation dataset (CV fold)  ----------
print(colored("\n[INFO]", "cyan"), "evaluating on validation set...")
lr_auc_val, lr_fpr_val, lr_tpr_val, thresholds_val, loss_val, preds_val=evaluate_model(x_val, y_val, model, threshold)
thresholds_val[0]=1

print(colored("\n[INFO]", "cyan"), "creating plots for validation set...")
create_plots(approach, lr_fpr_val, lr_tpr_val, thresholds_val, history, y_val, preds_val, rounds, "ROC_curve"+name_model+"VAL", "result_model"+name_model+"VAL")


#--------  generate the table of the final results  ----------
print("\nGenerating table of results...")
cprint("Copies \t fold \t ROUNDS\t threshold_VAL \t LOSS_val \t\t AUC_val", "yellow")
print(copies, "\t", fold , "\t", rounds, "\t", thresholds_val[np.argmin(abs(lr_tpr_val-(1-lr_fpr_val)))], "\t", loss_val, "\t", lr_auc_val)
with open(out_file, "a") as out:
    if lr_auc_val>0.5:
        out.write(str(thresholds_val[np.argmin(abs(lr_tpr_val-(1-lr_fpr_val)))])+ "\t"+str(loss_val)+ "\t"+ str(lr_auc_val)+"\n")