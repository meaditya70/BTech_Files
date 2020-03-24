import numpy as np
import random
from scipy import stats
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import matplotlib.font_manager



def processData(inputList,fname):
    mainList=[]
    nameList=[]
    output=[]
    desiredOutput=[]
    for i in range(0,len(inputList)-1):
        mainListPart=[]
        for j in range(0,len(inputList[i])):
            if(j==0):
                nameList.append(inputList[i][j])
            else:
                mainListPart.append(float(inputList[i][j]))
        mainList.append(mainListPart)

    mainList=[list(x) for x in zip(*mainList)]
    mainList=np.asarray(mainList, dtype=np.float64)

    output=inputList[-1]
    if(fname=="breast_preprocessed.txt"):
        for i in range(1,len(output)):
            if(output[i]=="luminal"):
                desiredOutput.append(1)
            else:
                desiredOutput.append(0)
    if(fname=="prostate_preprocessed.txt"):
        for i in range(1,len(output)):
            if(output[i]=="tumor"):
                desiredOutput.append(1)
            else:
                desiredOutput.append(0)
    if(fname=="lymphoma_preprocessed.txt"):
        for i in range(1,len(output)):
            if(output[i]=="1"):
                desiredOutput.append(1)
            else:
                desiredOutput.append(0)
    
    desiredOutput=np.asarray(desiredOutput, dtype=np.int64)
    return mainList,nameList,desiredOutput




def svmClassifier(x,y,t,o):
    clf = svm.SVC(kernel="linear",C=1,cache_size=1000)
    seed = len(x) - 1
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=kfold)
    clf.fit(x, y)
	
    obtainedOutput=[]
    obtainedOutput=clf.predict(t)
    count=0
    for i in range(0,len(obtainedOutput)):
    	if(obtainedOutput[i]==o[i]):
    		count=count+1
    total=len(obtainedOutput)       
    count=float(count)
    total=float(total)
    accuracy=count/total*100
    return scores.mean()



def pcaFeatures(X,y,nc):
	X_std = StandardScaler().fit_transform(X)
	sklearn_pca =  IncrementalPCA(n_components=nc)
	X_std=np.asarray(X_std, dtype=np.float64)
	Y_sklearn = sklearn_pca.fit_transform(X_std)
	return Y_sklearn

	
	
def outlierDet(mainList,desiredOutput,file_name):
	# Example settings
	rng = np.random.RandomState(42)
	outliers_fraction=0.261
	
	n_samples = len(desiredOutput)
	outlierInput=[]
	outlierOutput=[]
	X = mainList
	
	if(file_name=="prostate_preprocessed.txt"):
		classifiers = {"Local Outlier Factor": LocalOutlierFactor(n_neighbors=35,contamination=outliers_fraction)}
		for i, (clf_name, clf) in enumerate(classifiers.items()):
			y_pred = clf.fit_predict(X)
			print("Oulier Detection Method: Local Outlier Factor")
			print("Total No. of Samples:",len(mainList))
			i=0
			for indx in range(0,len(y_pred)):
				if(y_pred[indx]==1):
					outlierInput.append(mainList[indx])
					outlierOutput.append(desiredOutput[indx])
					i=i+1
			print("Total No. of Inliers:",i)
			print("Total No. of Outliers:",len(mainList)-i)
	
	if(file_name=="breast_preprocessed.txt"):
		classifiers = {"One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,kernel="linear", gamma=0.5)}
		for i, (clf_name, clf) in enumerate(classifiers.items()):
			if clf_name == "Local Outlier Factor":
				y_pred = clf.fit_predict(X)
			else:
				clf.fit(X)
				y_pred = clf.predict(X)
			print("Oulier Detection Method: One-Class SVM")
			
			print("Total No. of Samples:",len(mainList))
			i=0
			for indx in range(0,len(y_pred)):
				if(y_pred[indx]==1):
					outlierInput.append(mainList[indx])
					outlierOutput.append(desiredOutput[indx])
					i=i+1
			print("Total No. of Inliers:",i)
			print("Total No. of Outliers:",len(mainList)-i)
	
	return outlierInput,outlierOutput
	

	
	
#main begins here
print("\n Using IncrementalPCA with LinearSVM for all the 3 datasets \n #100% data for testing as well as training ")

maxC=21
file_array=["lymphoma_preprocessed.txt","prostate_preprocessed.txt","breast_preprocessed.txt"]

for file_name in file_array:

	print("\n\n",file_name)
	f = open(file_name,"r")
	d = f.readlines()
	inputList=[x.split(" ") for x in d]

	#preprocessing dataset
	mainList,nameList,desiredOutput=processData(inputList,file_name)
	mainList=np.asarray(mainList, dtype=np.float64)
	desiredOutput=np.asarray(desiredOutput, dtype=np.int64)
	
	
	cvArray=[]

	objects=[]
	
	#outlier Detection
	if(file_name!="lymphoma_preprocessed.txt"):
		mainList,desiredOutput=outlierDet(mainList,desiredOutput,file_name)
	else:
		print("Oulier Detection Method: None")

	#IncrementalPCA (choosing 2 to 20 Principal Components) + LinearSVM  
	for ncomp in range(2,maxC):
		objects.append(ncomp)
		lmainList=[]
		lmainList=pcaFeatures(mainList,desiredOutput,ncomp)

		#100% data for testing as well as training 
		trainIn=lmainList
		testIn=lmainList
		trainOut=desiredOutput
		testOut=desiredOutput
		
		cvArray.append(svmClassifier(trainIn,trainOut,testIn,testOut))
		
		
	maxCV=max(cvArray)
	maxIndex=cvArray.index(max(cvArray))+2

	print("Maximum CV Score: %0.2f, with %d Principal Components" %(maxCV,maxIndex))
	
	#plotting the SVM CV scores
	y_pos=np.arange(len(objects))
	performance=cvArray
	plt.bar(y_pos,performance,align='center', alpha=0.5)
	plt.xticks(y_pos,objects)
	plt.ylabel('Cross Validation Scores')
	plt.show()
	
	print("\n #############################################################################")
#endMain