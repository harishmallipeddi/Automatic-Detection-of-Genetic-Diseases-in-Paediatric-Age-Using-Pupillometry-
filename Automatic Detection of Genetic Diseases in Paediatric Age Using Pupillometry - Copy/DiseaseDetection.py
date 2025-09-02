from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import os
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout,Flatten
from sklearn.preprocessing import OneHotEncoder
import keras.layers
from sklearn.preprocessing import normalize

from keras.layers import Bidirectional

main = tkinter.Tk()
main.title("Automatic Detection of Genetic Diseases in Pediatric Age Using Pupillometry")
main.geometry("1300x1200")

global filename

global classifier
global left_X_train, left_X_test, left_y_train, left_y_test
global right_X_train, right_X_test, right_y_train, right_y_test

global left_X, left_Y

global left_pupil
global right_pupil
global count
global left
global right
global ids
global left_svm_acc
global right_svm_acc
global left_classifier
global right_classifier
global classifier
global ensemble_acc
global elm_acc
global lstm_acc,bilstm_acc

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Pupillometric  dataset loaded\n')

def filtering():
    global left_pupil
    global right_pupil
    global count
    global left
    global right
    global ids
    left_pupil = []
    right_pupil = []
    count = 0
    left = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    right = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    ids = 1
    for root, dirs, directory in os.walk('dataset'):
        for i in range(len(directory)):
            filedata = open('dataset/'+directory[i], 'r')
            lines = filedata.readlines()
            left_pupil.clear()
            right_pupil.clear()
            count = 0
            for line in lines:
                line = line.strip()
                arr = line.split("\t")
                if len(arr) == 8:
                    if arr[7] == '.....':
                        left_pupil.append(float(arr[3].strip()))
                        right_pupil.append(float(arr[6].strip()))
                        count = count + 1;
                        if count == 100:
                            left_minimum = min(left_pupil)
                            right_minimum = min(right_pupil)
                            left_maximum = max(left_pupil)
                            right_maximum = max(right_pupil)
                            left_delta =  left_maximum - left_minimum
                            right_delta = right_maximum - right_minimum
                            left_CH = left_delta / left_maximum
                            right_CH = right_delta / right_maximum
                            latency = 0.5
                            left_MCV = left_delta/(left_minimum - latency)
                            right_MCV = right_delta/(right_minimum - latency)
                            count = 0
                            left_pupil.clear()
                            right_pupil.clear()
                            if left_minimum > 500 and left_maximum > 500:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",1\n"
                            else:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",0\n"
                            if right_minimum > 500 and right_maximum > 500:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",1\n"
                            else:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",0\n"
                            ids = ids + 1
            filedata.close()
    
    text.delete('1.0', END)
    text.insert(END,'Features filteration process completed\n')
    text.insert(END,'Total patients found in dataset : '+str(ids)+"\n")
    
def featuresExtraction():
    f = open("left.txt", "w")
    f.write(left)
    f.close()
    f = open("right.txt", "w")
    f.write(right)
    f.close()
    text.delete('1.0', END)
    text.insert(END,'Both eye pupils extracted features saved inside left.txt and right.txt files \n')
    text.insert(END,"Extracted features are \nPatient ID, MAX, MIN, Delta, CH, Latency, MDV, CV and MCV\n")

def featuresReduction():
    text.delete('1.0', END)
    global left_X, left_Y
    global left_X_train, left_X_test, left_y_train, left_y_test
    global right_X_train, right_X_test, right_y_train, right_y_test
    left_pupil =  pd.read_csv('left.txt')
    right_pupil =  pd.read_csv('right.txt')
    cols = left_pupil.shape[1]

    left_X = left_pupil.values[:, 1:(cols-1)] 
    left_Y = left_pupil.values[:, (cols-1)]

    right_X = right_pupil.values[:, 1:(cols-1)] 
    right_Y = right_pupil.values[:, (cols-1)]

    indices = np.arange(left_X.shape[0])
    np.random.shuffle(indices)
    left_X = left_X[indices]
    left_Y = left_Y[indices]

    indices = np.arange(right_X.shape[0])
    np.random.shuffle(indices)
    right_X = right_X[indices]
    right_Y = right_Y[indices]

    left_X = normalize(left_X)
    right_X = normalize(right_X)
    

    left_X_train, left_X_test, left_y_train, left_y_test = train_test_split(left_X, left_Y, test_size = 0.2,random_state=42)
    right_X_train, right_X_test, right_y_train, right_y_test = train_test_split(right_X, right_Y, test_size = 0.2,random_state=42)

    text.insert(END,"Left pupil features training size : "+str(len(left_X_train))+" & testing size : "+str(len(left_X_test))+"\n")
    text.insert(END,"Right pupil features training size : "+str(len(right_X_train))+" & testing size : "+str(len(right_X_test))+"\n")

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Diameter')
    plt.plot(left_pupil['MAX'], 'ro-', color = 'indigo')
    plt.plot(right_pupil['MAX'], 'ro-', color = 'green')
    plt.legend(['Left Pupil', 'Right Pupil'], loc='upper left')
    plt.title('Pupil Diameter Graph')
    plt.show()   
    
    

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred     
    
def rightSVM():
    text.delete("1.0", END)

    global right_classifier
    text.delete('1.0', END)
    global right_svm_acc
    temp = []
    for i in range(len(right_y_test)):
        temp.append(right_y_test[i])
    temp = np.asarray(temp)    
    right_classifier = svm.SVC(C=80,kernel='rbf', class_weight='balanced', probability=True)
    right_classifier.fit(right_X_train, right_y_train)
    text.insert(END,"Right pupil SVM Prediction Results\n") 
    prediction_data = prediction(right_X_test, right_classifier) 
    right_svm_acc = accuracy_score(temp,prediction_data)*100
    right_svm_recall = recall_score(temp,prediction_data)*100
    right_svm_f1 = f1_score(temp,prediction_data)*100
    right_svm_precision = precision_score(temp,prediction_data)*100



    text.insert(END,"Right pupil SVM Accuracy : "+str(right_svm_acc)+"\n")
    text.insert(END,"Right pupil SVM recall_score : "+str(right_svm_recall)+"\n")
    text.insert(END,"Right pupil SVM f1_score : "+str(right_svm_f1)+"\n")
    text.insert(END,"Right pupil SVM precision_score : "+str(right_svm_precision)+"\n\n\n\n\n")


    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil SVM Algorithm Specificity : '+str(specificity)+"\n\n")

def leftSVM():
    text.delete("1.0", END)

    global left_classifier
    global left_svm_acc
    temp = []
    for i in range(len(left_y_test)):
        temp.append(left_y_test[i])
    temp = np.asarray(temp) 
    left_classifier = svm.SVC(C=80,kernel='rbf', class_weight='balanced', probability=True)
    left_classifier.fit(left_X_train, left_y_train)
    text.insert(END,"Left pupil SVM Prediction Results\n") 
    prediction_data = prediction(left_X_test, left_classifier) 
    left_svm_acc = accuracy_score(temp,prediction_data)*100
    left_svm_recall_score = recall_score(temp,prediction_data)*100
    left_svm_f1_score = f1_score(temp,prediction_data)*100
    left_svm_precision_score = precision_score(temp,prediction_data)*100


    text.insert(END,"Left pupil SVM Accuracy : "+str(left_svm_acc)+"\n")
    text.insert(END,"Left pupil SVM recall_score : "+str(left_svm_recall_score)+"\n")
    text.insert(END,"Left pupil SVM f1_score : "+str(left_svm_f1_score)+"\n")
    text.insert(END,"Left pupil SVM precision_score : "+str(left_svm_precision_score)+"\n\n\n")


    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Left pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Left pupil SVM Algorithm Specificity : '+str(specificity)+"\n\n")

def ensemble():
    text.delete("1.0", END)

    global classifier
    global ensemble_acc
    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))

    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    indices = np.arange(trainX.shape[0])
    np.random.shuffle(indices)
    trainX = trainX[indices]
    trainY = trainY[indices]

    left_classifier = svm.SVC(C=200,kernel='rbf', class_weight='balanced', probability=True)
    right_classifier = svm.SVC(C=200,kernel='rbf', class_weight='balanced', probability=True)

    temp = []
    for i in range(len(testY)):
        temp.append(testY[i])
    temp = np.asarray(temp) 

    classifier = VotingClassifier(estimators=[
         ('SVMLeft', left_classifier), ('SVMRight', right_classifier)], voting='hard')
    classifier.fit(trainX, trainY)
    text.insert(END,"Optimized Ensemble Prediction Results\n") 
    prediction_data = prediction(testX, classifier) 
    ensemble_acc =  (accuracy_score(temp,prediction_data)*100)
    ensemble_recall_score=  (recall_score(temp,prediction_data)*100)
    ensemble_f1_score =  (f1_score(temp,prediction_data)*100)
    ensemble_precision_score =  (precision_score(temp,prediction_data)*100)

    text.insert(END,"Ensemble OR Accuracy : "+str(ensemble_acc)+"\n")
    text.insert(END,"Ensemble OR recall_score : "+str(ensemble_recall_score)+"\n")
    text.insert(END,"Ensemble OR f1_score : "+str(ensemble_f1_score)+"\n")
    text.insert(END,"Ensemble OR precision_score : "+str(ensemble_precision_score)+"\n\n\n")


    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Specificity : '+str(specificity)+"\n\n")
def runBILSTM():
    text.delete("1.0", END)

    global bilstm_acc
    global left_X, left_Y
    Y = left_Y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y)
    X = left_X.reshape((left_X.shape[0], left_X.shape[1], 1))
    print(Y)
    print(X.shape)
    model = Sequential()
    model.add(Bidirectional(keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1))))
    model.add(Bidirectional(keras.layers.LSTM(32, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # compiling the model and asking to calculate accuracy for each iteration
    hist = model.fit(X, Y, verbose=2, batch_size=5, epochs=10)  # start training model with batch size 5 and epoch as 100 with X and Y input data
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    acc = accuracy * 100
    bilstm_acc = acc
    text.insert(END, "\nExtension BI-LSTM Accuracy : " + str(bilstm_acc) + "\n")
    text.insert(END, "Extension BI-LSTM Loss : " + str(loss) + "\n")
    # F1 Score, Recall, Precision
    y_pred = model.predict(X)
    temp = Y.argmax(axis=1)
    prediction_data = y_pred.argmax(axis=1)
    from sklearn.metrics import f1_score, recall_score, precision_score
    f1 = f1_score(temp, prediction_data)
    recall = recall_score(temp, prediction_data)
    precision = precision_score(temp, prediction_data)
    text.insert(END, "F1 Score : " + str(f1) + "\n")
    text.insert(END, "Recall : " + str(recall) + "\n")
    text.insert(END, "Precision : " + str(precision) + "\n")
    text.insert(END, '\nBi-LSTM Model Summary can be seen in black console for layer details\n')
    print(model.summary())
    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    text.insert(END, 'Right pupil Extension BI-LSTM Algorithm Sensitivity : ' + str(sensitivity) + "\n")
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    text.insert(END, 'Right pupil Extension BI-LSTM Algorithm Specificity : ' + str(specificity) + "\n")


def predict():
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "testData")
    test = pd.read_csv(filename)
    test = test.values[:, 0:7]
    total = len(test)
    text.insert(END,filename+" test file loaded\n");
    test = normalize(test)
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No disease detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Disease detected')+"\n\n")


def graph():
    height = [right_svm_acc,left_svm_acc,ensemble_acc,bilstm_acc]
    bars = ('Right Pupil SVM Acc','Left Pupil SVM Acc','Ensemble OR (L & R Pupil) Acc','Extension BI-LSTM Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms Accuracy Comparison Graph")
    plt.show()
    
font = ('times', 16, 'bold')
title = Label(main, text='Automatic Detection of Genetic Diseases in Pediatric Age Using Pupillometry')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Pupillometric Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

filterButton = Button(main, text="Run Filtering", command=filtering)
filterButton.place(x=700,y=200)
filterButton.config(font=font1) 

extractButton = Button(main, text="Run Features Extraction", command=featuresExtraction)
extractButton.place(x=700,y=250)
extractButton.config(font=font1) 

featuresButton = Button(main, text="Run Features Reduction", command=featuresReduction)
featuresButton.place(x=700,y=300)
featuresButton.config(font=font1)

rightsvmButton = Button(main, text="Run SVM on Right Eye Features", command=rightSVM)
rightsvmButton.place(x=700,y=350)
rightsvmButton.config(font=font1)

leftsvmButton = Button(main, text="Run SVM on Left Eye Features", command=leftSVM)
leftsvmButton.place(x=700,y=400)
leftsvmButton.config(font=font1)


ensembleButton = Button(main, text="Run OR Ensemble Algorithm (Left & Right SVM)", command=ensemble)
ensembleButton.place(x=700,y=450)
ensembleButton.config(font=font1)

bilstmButton = Button(main, text="Run Extension BILSTM", command=runBILSTM)
bilstmButton.place(x=700,y=500)
bilstmButton.config(font=font1)


graphButton = Button(main, text="Accuracy Graph with Metrics", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)


predictButton = Button(main, text="Predict Disease", command=predict)
predictButton.place(x=700,y=600)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
