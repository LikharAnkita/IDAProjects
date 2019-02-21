#importing all the required libraries

from __future__ import print_function
import pydotplus
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score,classification_report, confusion_matrix, recall_score, precision_score
from sklearn.tree import export_graphviz
from sklearn import tree, preprocessing
from subprocess import call
from sklearn.cross_validation import cross_val_score
from pprint import pprint
from sklearn.grid_search import RandomizedSearchCV
import os
from sklearn.svm import SVC  
import subprocess
from time import time
from operator import itemgetter
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_extraction import DictVectorizer


# =====================================================================
#Function for downloading the data
def download_data(fileName):
    frame = pd.read_excel(fileName, sheet_name='Sheet1',header=None)
    return frame
# =====================================================================
#Funtion for splitting and saving the data
def get_features_and_labels(frame):
    
    arr = np.array(frame)
    # Use the last column as the target value
    X, Y, labels = arr[:, :-2], arr[:, 1], arr[:, -1]
    return X, Y, labels
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
def run_randomsearch(X, y, clf, param_dist, cv=5,
                     n_iter_search=20):
    """Run a random search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    param_dist -- [dict] list, distributions of parameters
                  to sample
    cv -- fold of cross-validation, default 5
    n_iter_search -- number of random parameter sets to try,
                     default 20.
    Returns
    -------
    top_params -- [dict] from report()
    """
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,cv=cv,
                                       n_iter=n_iter_search)
    start = time()
    random_search.fit(X, y)
    print(("\nRandomizedSearchCV took {:.2f} seconds "
           "for {:d} candidates parameter "
           "settings.").format((time() - start),
                               n_iter_search))
    top_params = report(random_search.grid_scores_, 50)
    return top_params


def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.
    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models
    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std Deviation: {1:.3f})" " Cross Validation score: {2}").format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores),
                   score.cv_validation_scores))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

#=====================================================================
def visualize_tree(treeClassifier, feature_names, fn="dt"):
    """Create tree png using graphviz.
    """
    dotfile = fn + ".dot"
    pngFile = fn +".png"
    tree.export_graphviz(treeClassifier, out_file='tree.dot', feature_names=feature_names,filled=True,rounded=True)
    call(['dot', '-T', 'png', 'tree.dot', '-o', pngFile], shell = True)
    return None
#======================================================================
#Main method
if __name__ == '__main__':
    # Download the data set from URL
    #print("Downloading data from {}".format(URL))
    #====Ques 1==========
    frame = download_data('HW2-Synth-Data.xls')
    X, Y, labels = get_features_and_labels(frame)
    print(labels)
    color= ['red' if l == 0 else 'green' for l in labels]
    plt.scatter(X, Y, color=color)
    plt.show();
#=================ques 1(b)===============================================
headers =["X","Y","class"]
features = ["X","Y"]
df = pd.read_excel('HW2-Synth-Data.xls', sheet_name='Sheet1',header=None)
arr2= np.array(df)
X, y = arr2[:, :-1], arr2[:, -1]
# dict of parameter list/distributions to sample
param_dist = {"criterion": ["gini", "entropy"],
                  "min_samples_split": randint(2, 20),
                  "max_depth": randint(5, 10),
                  "min_samples_leaf": randint(1, 20),
                  "max_leaf_nodes": randint(10, 20)}
dt = DecisionTreeClassifier()
ts_rs = run_randomsearch(X, y, dt, param_dist, cv=5,
                             n_iter_search=5)
print("\nts_rs\n")
print(ts_rs)
print("\n-- best parameters:")
for k, v in ts_rs.items():
      print("parameters: {:<20s} setting: {}".format(k, v))

## test the retuned best parameters
print("\n\n-- Testing best parameters [Random]...")
dt_ts_rs = DecisionTreeClassifier(**ts_rs)
scores = cross_val_score(dt_ts_rs, X, y, cv=5)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                              scores.std()))
dt_ts_rs.fit(X, y)
visualize_tree(dt_ts_rs, features, fn="rand_best")
    # predict the result
y_pred = dt_ts_rs.predict(X) 
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))  
print()
print('Accuracy score: ')
print(accuracy_score(y,y_pred))
print()
print('Precision:')
classprecision =precision_score(y,y_pred,labels=None, average=None)
print(classprecision)
print()
print("Precision for Class label 0: {:.5f}".format(classprecision[0]))
print("Precision for Class label 1: {:.5f}".format(classprecision[1]))
print()
print('Recall')
classRecall = recall_score(y,y_pred,labels=None, average=None)
print(classRecall)
print()
print("Recall for Class label 0: {:.5f}".format(classRecall[0]))
print("Recall for Class label 1: {:.5f}".format(classRecall[1]))


#=====ques 1(c)=================

plot_colors = "ryb"
plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
Z = dt_ts_rs.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.xlabel("X")
plt.ylabel("Y")
headings= ["Class0","Class1"]
    # Plot the training points
for i, color in zip(range(2), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=headings[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()
#=================ques 1(e)============
param_dist = {"kernel": ["linear"],
              "random_state":[6,7,10],
              "max_iter":[-1,3],
              'C': [1, 10, 100, 1000],
              'gamma': [1e-3, 1e-4]}
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(SVC(), param_dist, cv=5,
                       scoring='%s' % score)
    clf.fit(X, y)
    top_params = report(clf.grid_scores_, 250)
    print("Best parameters set found on development set:")
    print()
    print(top_params)
    print()
    print("Grid scores on development set:")
    print()
    for k, v in top_params.items():
      print("parameters: {:<20s} setting: {}".format(k, v))

# test the retuned best parameters
print("\n\n-- Testing best parameters [Random]...")
dt_ts_rs = SVC(**top_params)
dt_ts_rs.fit(X, y)
y_pred = dt_ts_rs.predict(X) 
scores = cross_val_score(dt_ts_rs, X, y, cv=5)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                              scores.std()))
print()
print("Detailed classification report:")
print()
print(classification_report(y, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))  
print()
print('Accuracy:')
print(accuracy_score(y,y_pred))
print()
print('Precision:')
classprecision =precision_score(y,y_pred,labels=None, average=None)
print(classprecision)
print()
print("Precision for Class label 0: {:.5f}".format(classprecision[0]))
print("Precision for Class label 1: {:.5f}".format(classprecision[1]))
print()
print('Recall')
classRecall = recall_score(y,y_pred,labels=None, average=None)
print(classRecall)
print()
print("Recall for Class label 0: {:.5f}".format(classRecall[0]))
print("Recall for Class label 1: {:.5f}".format(classRecall[1]))

titles = ['SVC with linear kernel']
#plotting the SVM boundary and vectors
X0, X1 = X[:, 0], X[:, 1]
h=0.2
plt.scatter(X0, X1, c=y, cmap=plt.cm.Paired, s=30)
ax =plt.gca()
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = np.linspace(y_min, y_max, 100)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = dt_ts_rs.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(dt_ts_rs.support_vectors_[:, 0],
                   dt_ts_rs.support_vectors_[:, 1],
                   s=100, linewidth=3, facecolors='none',edgecolors='k');
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(titles[0])
plt.show()


#=========================1(f)========
# Set the parameters by cross-validation
param_dist = {"kernel": ["rbf"],
              "max_iter":[-1,3],
               'C': [0.1,1,10,100,1000],
              'gamma': [1,0.1,0.01,0.001,0.0001]}
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(SVC(), param_dist, cv=5,verbose=3,scoring='%s' % score)
    clf.fit(X, y)
    top_params = report(clf.grid_scores_, 250)
    print("Best parameters set found on development set:")
    print()
    print(top_params)
    print()
    print("Grid scores on development set:")
    print()
    for k, v in top_params.items():
      print("parameters: {:<20s} setting: {}".format(k, v))

# test the retuned best parameters
print("\n\n-- Testing best parameters [Random]...")
dt_ts_rs = SVC(**top_params)
dt_ts_rs.fit(X, y)
y_pred = dt_ts_rs.predict(X) 
scores = cross_val_score(dt_ts_rs, X, y, cv=5)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                              scores.std()))
print()
print("Detailed classification report:")
print(classification_report(y, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))  
print()
print('Accuracy:')
print(accuracy_score(y,y_pred))
print()
print('Precision:')
classprecision =precision_score(y,y_pred,labels=None, average=None)
print(classprecision)
print()
print("Precision for Class label 0: {:.5f}".format(classprecision[0]))
print("Precision for Class label 1: {:.5f}".format(classprecision[1]))
print()
print('Recall')
classRecall = recall_score(y,y_pred,labels=None, average=None)
print(classRecall)
print()
print("Recall for Class label 0: {:.5f}".format(classRecall[0]))
print("Recall for Class label 1: {:.5f}".format(classRecall[1]))
titles = ('SVC with rbf kernel')

# Set-up 2x2 grid for plotting.
titles =['SVC with RBF kernel']
X0, X1 = X[:, 0], X[:, 1]
h=0.2
plt.scatter(X0, X1, c=y, cmap=plt.cm.Paired, s=30)
ax =plt.gca()
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = np.linspace(y_min, y_max, 100)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = dt_ts_rs.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(dt_ts_rs.support_vectors_[:, 0],
                   dt_ts_rs.support_vectors_[:, 1],
                   s=100, linewidth=3, facecolors='none',edgecolors='k');
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(titles[0])
plt.show()



    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean, std * 2, params))
   
# x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.4,random_state=0)


#    dataFiles = []
#    dataFiles.append('Biomechanical_Data_column_2C_weka.csv')
#    dataFiles.append('BiomechanicalData_column_3C_weka.csv')
#    dataFiles.append('Biomechanical_Data_column_2C_weka.csv')
#    min_Nodes_Array = []
    
#    data_feature_names =[ 'pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis' ]
#    data_file_being_processed =1
#for files in dataFiles:
#    frame = download_data(files)
#    ## Process data into feature and label arrays
#    #print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
#    print("Solution starts for question Number: " +str(data_file_being_processed))
#    print("taking Data from file:" +files)
#    if (data_file_being_processed==3): 
#        for i in range(0,6):
#            print("Taking column number: " +str(i)+ " Column name is: " +data_feature_names[i])
#            frame.sort_values(data_feature_names[i], inplace=True)
#            X_train, X_test, y_train, y_test = split_column_values(frame,data_feature_names[i],i)
#            print("X - Test")
#            print(X_test)
#            print('X - Train')
#            print(X_train)
#            print("y - Test")
#            print(y_test)
#            print('y - T
#            rain')
#            print(y_train)
#            min_Nodes_Array = [8,12]
#            print("Building decision trees with Min leaf node as: 8, 12")
#            Create_Decision_Tree_For_All_Nodes(X_train, X_test, y_train, y_test,"{}{}".format("Column_",data_feature_names[i]),min_Nodes_Array,files)
            
#    else:
#            X_train, X_test, y_train, y_test = get_features_and_labels(frame,files)
#            min_Nodes_Array= [3,8,12,30,50]
#            print("X - Test")
#            print(X_test)
#            print('X - Train')
#            print(X_train)
#            print("y - Test")
#            print(y_test)
#            print('y - Train')
#            print(y_train)
#            print("Building decision trees with Min leaf node as: 3, 8, 12, 30,50")
#            Create_Decision_Tree_For_All_Nodes(X_train, X_test, y_train, y_test,None,min_Nodes_Array,files)
#    data_file_being_processed = data_file_being_processed +1
#    ## Evaluate multiple classifiers on the data
#    print("Evaluating classifiers")
#dt_old = DecisionTreeClassifier(min_samples_split=20, random_state=31)
#dt_old.fit(X, y)
#scores = cross_val_score(dt_old, X, y, cv=5)
#visualize_tree(dt_old, features, fn="old_classifier")
#print("Decision Tree mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()))
##visualize_tree(dt, features)
#y_pred = dt_old.predict(X)
#print('Accuracy score: ')
#print(accuracy_score(y,y_pred))


    

#=================================================
#Assignment Question 3
 