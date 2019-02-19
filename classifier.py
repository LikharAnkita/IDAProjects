#importing all the required libraries
import pydotplus
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score,classification_report, confusion_matrix, recall_score, precision_score
from sklearn.tree import export_graphviz
from sklearn import tree, preprocessing
from subprocess import call


# =====================================================================
#Function for downloading the data
def download_data(fileName):
    frame = pd.read_csv(fileName)
    return frame
# =====================================================================
#Funtion for splitting and saving the data
def get_features_and_labels(frame, fileName):
    
    arr = np.array(frame)
    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    np.set_printoptions(suppress=True,
    formatter={'float_kind':'{:16.10f}'.format}, linewidth=130)
    
    # Use 80% of the data for training; test against the rest
    
    X_train, X_test , y_train, y_test= train_test_split(X, y,random_state =11, test_size=0.32)
    if '3C'in fileName: 
        fileName= 'Data_3C'
    else:
        fileName = 'Data_2C'
    np.savetxt("{}{}".format(fileName,"_X_test.csv"), X_test,delimiter=",")
    np.savetxt("{}{}".format(fileName,"_y_test.csv"), y_test, fmt ='%s',delimiter=",")
    np.savetxt("{}{}".format(fileName,"_X_train.csv"), X_train,delimiter=",")
    np.savetxt("{}{}".format(fileName,"_y_train.csv"), y_train,fmt ='%s',delimiter=",")
    # Return the training and test sets
    return X_train, X_test, y_train, y_test

# =====================================================================
#Function for creating the Decision Tree
def evaluate_classifier(X_train, X_test, y_train, y_test, leaf_Value):

    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=leaf_Value)
    clf = clf.fit(X_train, y_train)
    return clf
# =====================================================================
#Function for plotting Accuracy,Precision and Recall curves
def plot(accuracy_allArr,precision_score_All,recall_score_all,min_Nodes_Array,files,columns):
    
    if ('3C' in files and columns is None):
        pngFileAbbreviation = 'Plot_BiomechanicalData_column_3C_weka'
    elif columns is None:
        pngFileAbbreviation = 'Plot_Biomechanical_Data_column_2C_weka'
    else:
        pngFileAbbreviation = 'Plot_Biomechanical_Data_column_2C_weka_Spliting_Columns'+ columns
    #Display the results
    print("Plotting the results")
    plt.title("Accuracy score against minimum no. of leaf nodes in tree")
    plt.xlabel("Minimum number of leaf nodes")
    plt.ylabel("Accuracy")
    plt.plot(min_Nodes_Array,accuracy_allArr, label="Accuracy score", color="yellow" )
    plt.tight_layout()
    plt.legend(loc="best")
    print("Accuracy plot saved in {}{}.png".format(pngFileAbbreviation,'_Accuracy_score_plot'))
    plt.savefig("{}{}.png".format(pngFileAbbreviation,'_Accuracy_score_plot'))
    plt.show();
    
    plt.title("Precision score against minimum no. of leaf nodes in tree")
    plt.xlabel("Minimum number of leaf nodes")
    plt.ylabel("Precision")
    plt.plot(min_Nodes_Array,precision_score_All, label="Precision Score", color="Red" )
    plt.tight_layout()
    plt.legend(loc="best")
    print("Precision Plot saved in {}{}.png".format(pngFileAbbreviation,'_Precision_score_plot'))
    plt.savefig("{}{}.png".format(pngFileAbbreviation,'_Precision_score_plot'))
    plt.show();
    

    plt.title("Recall score against minimum no. of leaf nodes in tree")
    plt.xlabel("Minimum number of leaf nodes")
    plt.ylabel("Recall")
    plt.plot(min_Nodes_Array,recall_score_all, label="Recall score", color="Blue" )
    plt.tight_layout()
    plt.legend(loc="best")
    print("Recall plot saved in {}{}.png".format(pngFileAbbreviation,'_Recall_score_plot'))
    plt.savefig("{}{}.png".format(pngFileAbbreviation,'_Recall_score_plot'))
    plt.show();
    
    return None
#======================================================================
#Function for cross-Validating test data
def preditValues(clfs):
    y_pred = clfs.predict(X_test) 
    return y_pred
#========================================================================
def Create_Decision_Tree_For_All_Nodes(X_train, X_test, y_train, y_test, tree_name, min_Nodes_Array,files):
    
    accuracy_allArr = []
    precision_score_All =[]
    recall_score_all=[]
    for elemVal in min_Nodes_Array:
            print("Calculation for Minimum Leaf Nodes= "+str(elemVal))
            classifier1 =  evaluate_classifier(X_train, X_test, y_train, y_test, elemVal)
            y_pred = preditValues(classifier1)
            df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
            #Printing Confusion matrix
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))  
            #Printing Classification Matrix
            #print(classification_report(y_test, y_pred))  
            accuracy_allArr.append(accuracy_score(y_test,y_pred))
            if '3C' in files:
                precision_score_All.append(precision_score(y_test,y_pred,labels=None, average=None))
                print('precision')
                print(precision_score(y_test,y_pred,labels=None, average=None))
                print('recall')
                recall_score_all.append(recall_score(y_test,y_pred,labels=None, average=None))
                print(recall_score(y_test,y_pred,labels=None, average=None))
                print('Accuracy')
                print(accuracy_score(y_test,y_pred))
                if tree_name is None:
                    pngFile = "{}{}{}.png".format("Tree_With_",elemVal,'_Node_BiomechanicalData_column_3C_weka')
                else:
                    pngFile = "{}{}{}{}.png".format("Tree_With_",tree_name,elemVal,'_Node_BiomechanicalData_column_3C_weka')
                print("Decision Tree with " +str(elemVal)+ "Nodes is saved in: " +pngFile)
            else:
                precision_score_All.append(precision_score(y_test,y_pred,labels=None, pos_label='Normal'))
                print('precision')
                print(precision_score(y_test,y_pred,labels=None, pos_label='Normal'))
                print('recall')
                print(recall_score(y_test,y_pred,labels=None, pos_label='Normal'))
                print('Accuracy')
                print(accuracy_score(y_test,y_pred))
                recall_score_all.append(recall_score(y_test,y_pred,labels=None, pos_label='Normal'))
                if tree_name is None:
                    pngFile = "{}{}{}.png".format("Tree_With_",elemVal,'_Node_Biomechanical_Data_column_2C_weka')
                else:
                    pngFile = "{}{}{}{}.png".format("Tree_With_",tree_name,elemVal,'_Node_Biomechanical_Data_column_2C_weka')
                print("Decision Tree with " +str(elemVal)+ "Nodes is saved in: " +pngFile)
            
            #listForPlot.append(provideComparisonValues(classifier1, X_test,y_test,y_pred, "{}{}".format("plotvalue",elemVal)))
            tree.export_graphviz(classifier1, out_file='tree.dot', feature_names=data_feature_names,filled=True,rounded=True)
            call(['dot', '-T', 'png', 'tree.dot', '-o', pngFile], shell = True)

    print("For min_leaf nodes 3,8,12,30,50")
    print('Accuracy')

    for nodes in range(0,len(min_Nodes_Array)):
        print(accuracy_allArr[nodes])

    print('Precision')
    for nodes in range(0,len(min_Nodes_Array)):
        print(precision_score_All[nodes])

    print('Recall')
    for nodes in range(0,len(min_Nodes_Array)):
        print(recall_score_all[nodes])
    if tree_name is None:
        plot(accuracy_allArr,precision_score_All,recall_score_all,min_Nodes_Array,files,None)
    else:
        plot(accuracy_allArr,precision_score_All,recall_score_all,min_Nodes_Array,files,tree_name)
    return None 

# =====================================================================
def split_column_values(frame,feature_name, index_val):
   
    arr = np.array(frame[feature_name])
    arr2 = np.array(frame)
    print(arr)
    max_arr= arr.max()
    min_arr= arr.min()
    
    max_Val = float(max_arr) 
    min_Val = float(min_arr)
    print(max_arr);
    print(min_arr);
    avg = (max_Val+min_Val)/4
    avg2 = avg*2
    avg3 = avg * 3
    for val in arr2:
        if val[index_val] < avg:
            val[index_val]=0
        elif (val[index_val] <avg2 and val[index_val]>=avg):
            val[index_val]=1
        elif(val[index_val] >= avg2 and val[index_val] < avg3):
            val[index_val]=2
        else:
            val[index_val]=3

    print("Boundaries for division of column attribute:" +feature_name)
    print("1st Interval which is replaced by 0:" +str(min_Val)+ " to "+str(avg))
    print("2nd Interval which is replaced by 1:" +str(avg)+ " to "+ str(avg2))
    print("3rd Interval which is replaced by 2:" +str(avg2)+ " to "+ str(avg3))
    print("last Interval which is replaced by 3:" +str(avg3)+ " to "+ str(max_Val))
    for val in arr2:
        if(val[index_val]==3):
            print(val)
    X, y = arr2[:, :-1], arr2[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    np.set_printoptions(suppress=True,
    formatter={'float_kind':'{:16.10f}'.format}, linewidth=130)
    X_train, X_test , y_train, y_test= train_test_split(X, y,random_state =23, test_size=0.32)
    np.savetxt("{}{}".format(feature_name,"_X_test.csv"), X_test,delimiter=",")
    np.savetxt("{}{}".format(feature_name,"_y_test.csv"), y_test, fmt ='%s',delimiter=",")
    np.savetxt("{}{}".format(feature_name,"_X_train.csv"), X_train,delimiter=",")
    np.savetxt("{}{}".format(feature_name,"_y_train.csv"), y_train,fmt ='%s',delimiter=",")
    return X_train, X_test, y_train, y_test
#======================================================================
#Main method
if __name__ == '__main__':
    # Download the data set from URL
    #print("Downloading data from {}".format(URL))
    dataFiles = []
    dataFiles.append('Biomechanical_Data_column_2C_weka.csv')
    dataFiles.append('BiomechanicalData_column_3C_weka.csv')
    dataFiles.append('Biomechanical_Data_column_2C_weka.csv')
    min_Nodes_Array = []
    
    data_feature_names =[ 'pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis' ]
    data_file_being_processed =1
for files in dataFiles:
    frame = download_data(files)
    ## Process data into feature and label arrays
    #print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    print("Solution starts for question Number: " +str(data_file_being_processed))
    print("taking Data from file:" +files)
    if (data_file_being_processed==3): 
        for i in range(0,6):
            print("Taking column number: " +str(i)+ " Column name is: " +data_feature_names[i])
            frame.sort_values(data_feature_names[i], inplace=True)
            X_train, X_test, y_train, y_test = split_column_values(frame,data_feature_names[i],i)
            print("X - Test")
            print(X_test)
            print('X - Train')
            print(X_train)
            print("y - Test")
            print(y_test)
            print('y - Train')
            print(y_train)
            min_Nodes_Array = [8,12]
            print("Building decision trees with Min leaf node as: 8, 12")
            Create_Decision_Tree_For_All_Nodes(X_train, X_test, y_train, y_test,"{}{}".format("Column_",data_feature_names[i]),min_Nodes_Array,files)
            
    else:
            X_train, X_test, y_train, y_test = get_features_and_labels(frame,files)
            min_Nodes_Array= [3,8,12,30,50]
            print("X - Test")
            print(X_test)
            print('X - Train')
            print(X_train)
            print("y - Test")
            print(y_test)
            print('y - Train')
            print(y_train)
            print("Building decision trees with Min leaf node as: 3, 8, 12, 30,50")
            Create_Decision_Tree_For_All_Nodes(X_train, X_test, y_train, y_test,None,min_Nodes_Array,files)
    data_file_being_processed = data_file_being_processed +1
    ## Evaluate multiple classifiers on the data
    print("Evaluating classifiers")

    

#=================================================
#Assignment Question 3
 