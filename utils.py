# LIBRERIAS
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import save_img
from sklearn.manifold import TSNE
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model

# Plot and save learning curves
def plot_learning_curves(H, folder_to_save, name_to_save, partition, curve):
    '''
    This functions stores the learning curves performed during the training stage enclosed in H
    '''

    if curve=='acc' or curve==None:
        # Save curves training
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history['categorical_accuracy'])
        plt.plot(H.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', partition], loc='upper left')
        plt.savefig(folder_to_save + '/Accuracy_' + name_to_save + '.png')

    if curve=='loss' or curve==None:
        plt.figure()
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', partition], loc='upper left')
        plt.savefig(folder_to_save + '/Loss_' + name_to_save + '.png')

# Analysis of the latent space
def extract_TSNE(centroids, val_features):

    # SAVE CENTROIDS
    centrs = np.empty((len(centroids), np.shape(centroids[0])[0]))
    centrs[0] = centroids[0]
    centrs[1] = centroids[1]
    centrs[2] = centroids[2]

    # CONCATENATE DATA
    tsne_vector = np.concatenate((centrs, val_features))

    # Define tsne parameters. By default: 2, 30, 12, 200, 1000, euclidean, barnes_hut
    # metrics = euclidean, cosine, cityblock, canberra, etc
    # method = barnes_hut, exact
    space, perp, early_ex, lr, n_iter, metric, method = 2, 10, 12, 50, 2000, 'euclidean', 'barnes_hut' # barnes_hut or exact
    tsne = TSNE(n_components=space, perplexity=perp, early_exaggeration=early_ex, learning_rate=lr,
                   n_iter=n_iter, metric=metric, method=method).fit_transform(tsne_vector)
    return tsne

# Extract classification results
def report_classification_results(y_true, y_pred, classes, flag_ret, class_mean):
    # Extract confusion matrix
    conf_mat = metrics.confusion_matrix(y_true, y_pred)

    # Compute metrics
    headers = []
    g_TP, g_TN, g_FN, g_FP = [],[],[],[]
    ss, ee, pp, nn, ff, aa = [],[],[],[],[],[]
    S, E, PPV, NPV, F, ACC = ['Sensitivity'], ['Specificity'], ['PPV (precision)'], ['NPV'], ['F1-score'], ['Accuracy']
    for i in range(len(classes)):
        headers.append(classes[str(i)])

        # Extract indicators per class
        TP = conf_mat[i,i]
        TN = sum(sum(conf_mat))-sum(conf_mat[i,:])-sum(conf_mat[:,i])+TP
        FN = sum(conf_mat[i,:])-conf_mat[i,i]
        FP = sum(conf_mat[:,i])-conf_mat[i,i]

        # Extract metrics per class
        sen = TP/(TP+FN)
        spe = TN/(TN+FP)
        ppv = TP/(TP+FP)
        npv = TN/(TN+FN)
        f1_s = 2*ppv*sen/(ppv+sen)
        acc = (TP+TN)/(TP+TN+FP+FN)

        # Create table for printing
        S.append(str(round(sen,4)))
        E.append(str(round(spe,4)))
        PPV.append(str(round(ppv,4)))
        NPV.append(str(round(npv,4)))
        F.append(str(round(f1_s,4)))
        ACC.append(str(round(acc,4)))

        # Global indicators
        g_TP.append(TP), g_TN.append(TN), g_FP.append(FP), g_FN.append(FN)
        # Global metrics
        ss.append(sen), ee.append(spe), pp.append(ppv), nn.append(npv), ff.append(f1_s), aa.append(acc)

    # Extract micro values
    micro_S =  round(sum(g_TP)/(sum(g_TP)+ sum(g_FN)),4)
    micro_E = round(sum(g_TN)/(sum(g_TN)+sum(g_FP)),4)
    micro_PPV = round(sum(g_TP)/(sum(g_TP)+sum(g_FP)),4)
    micro_NPV = round(sum(g_TN)/(sum(g_TN)+sum(g_FN)),4)
    micro_F1 = round(2*micro_S*micro_PPV/(micro_S+micro_PPV),4)
    micro_Acc = round((sum(g_TP)+sum(g_TN))/(sum(g_TP)+sum(g_TN)+sum(g_FP)+sum(g_FN)),4)
    # Extract macro values
    macro_S = round(sum(ss)/len(ss),4)
    macro_E = round(sum(ee)/len(ee),4)
    macro_PPV = round(sum(pp)/len(pp),4)
    macro_NPV = round(sum(nn)/len(nn),4)
    macro_F1 = round(sum(ff)/len(ff),4)
    macro_Acc = round(sum(aa)/len(aa),4)

    # Construct the table
    S.append('---'), E.append('---'), PPV.append('---'), NPV.append('---'), F.append('---'), ACC.append('---')
    S.append(micro_S), E.append(micro_E), PPV.append(micro_PPV), NPV.append(micro_NPV), F.append(micro_F1), ACC.append(micro_Acc)
    S.append(macro_S), E.append(macro_E), PPV.append(macro_PPV), NPV.append(macro_NPV), F.append(macro_F1), ACC.append(macro_Acc)

    # Define table
    my_data = [tuple(S), tuple(E), tuple(PPV), tuple(NPV), tuple(F), tuple(ACC)]

    # MODEL metrics
    auc = metrics.roc_auc_score(y_true=to_categorical(y_true, num_classes=len(classes)), y_score=to_categorical(y_pred, num_classes=len(classes)))

    # Printing results
    headers.append('-'), headers.append('micro-Avg'), headers.append('macro-Avg')
    print(tabulate(my_data, headers=headers))
    print('------------------------\nAUC', round(auc,4))

    if flag_ret==True:
        results = {'S': S[class_mean],
                   'E': E[class_mean],
                   'PPV': PPV[class_mean],
                   'NPV': NPV[class_mean],
                   'F1': F[class_mean],
                   'ACC': ACC[class_mean],
                   'AUC': auc}
        return results

# Print and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, dir_out = '', flag_save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if len(classes)==2:
        classes = [classes['0'], classes['1']]
    elif len(classes)>2:
        classes = [classes['0'], classes['1'], classes['2']]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if flag_save==True:
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        ax.figure.savefig(dir_out)
        plt.close()

        return ax

# Extract class activation maps
def CAMs_computation(folders, sev_opt, idx, model, layerToPredict, folderToSave, folder_csv_val):

    for k in np.arange(0, len(folder_csv_val)):
        img = cv2.imread(folders['data_DB_2'] + folder_csv_val.ID[k])
        img = cv2.resize(img, (sev_opt['y'], sev_opt['x']))
        img = img / 255.0
        image = np.expand_dims(img, axis=0)

        # índice de la clase predicha
        #idx = np.argmax(predictions, axis=1)

        gradModel = Model(inputs=model.input, outputs=[layerToPredict.output, model.output[0]])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOut, preds) = gradModel(inputs)
            loss = preds[:, idx[k]]

        grads = tape.gradient(loss, convOut)

        castConvOut = tf.cast(convOut > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOut * castGrads * grads
        convOut = convOut[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOut), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 1e-8
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        cm = plt.get_cmap('jet')
        colored_image = cm(heatmap)[:, :, :3]
        cam_img = img * 0.8 + colored_image * 0.2

        ### SAVE
        # predictions
        y_pred = idx[k]
        if y_pred==0:
            pred_label = 'Healthy'
        elif y_pred==1:
            pred_label = 'Low'
        elif y_pred==2:
            pred_label = 'High'

        # Ground truth
        label = folder_csv_val.Grade[k]

        saveDir = folderToSave + '/' + 'gt_' + label + '_pred_' + pred_label
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        save_img(saveDir + '/' + folder_csv_val.ID[k], cam_img)