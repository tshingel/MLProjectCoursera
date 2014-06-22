Machine Learning Algorithm to Predict Activity Quality From Activity Monitors
========================================================
# Human Activity Recognition
Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify *how much* of a particular activity they do, but they rarely quantify *how well* they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is availalbe from the [website](http://groupware.les.inf.puc-rio.br/har). 

## Data 

The training data for the project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).

The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

# Data Pre-Processing 
 
We start by loading the training data into our workspace: 

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)
pmlData <- read.csv("./train/pml-training.csv", stringsAsFactors = FALSE)
```

The dimentions of **pmlData** data frame are:

```r
dim(pmlData)
```

```
## [1] 19622   160
```

Upon inspection of **pmlData** date frame using function **str(pmlData)**, we can see that the measurements data spans columns 8 through 160. Therefore, we remove the first 7 columns from our data frame.


```r
pmlData <- pmlData[, -c(1:7)] 
```

Next, we create data partitioning by splitting the data into training and testing data sets.

```r
set.seed(1)
inTrain <- createDataPartition(pmlData$classe, p = 3/4)[[1]]
 
trainData <- pmlData[inTrain,]
testData <- pmlData[-inTrain,]
```
In what follows, we use **trainData** for training and **testData** for cross-validation.
To reduce the amount of time needed for training, we further reduce the number of samples in **trainData**. As we will see later, the resulting model will still have a high accuracy. 


```r
set.seed(2)
idx <- createDataPartition(trainData$classe, p = 1/2)[[1]]
trainData <- trainData[idx,]
nrow(trainData)
```

```
## [1] 7360
```
Remove **classe** column from data and convert it to a factor variable. 


```r
trainclass <- factor(trainData$classe)
trainData <- trainData[, colnames(trainData) != "classe"]
testclass <- factor(testData$classe)
testData <- testData[, colnames(testData) != "classe"]
```

The data contains numeric predictor variables saved as characters. In order to apply **train** function, we need to convert these fields to numeric types. This is done as outlined below.    


```r
options(warn = -1)
charcols <- sapply(trainData, is.character)
## convert character columns to numeric 
tmp.train <- apply(trainData[,charcols], 2, function(x) as.numeric(x))   
tmp.train <- as.data.frame(tmp.train)
tmp.test <- apply(testData[,charcols], 2, function(x) as.numeric(x))
tmp.test <- as.data.frame(tmp.test)
## update data frames 
trainData[,charcols] <- tmp.train
testData[,charcols] <- tmp.test 
```

We remove predictors which have $80\%$ and more missing values

```r
obs <- nrow(trainData)
nacols <- apply(trainData, 2, function(x) {sum(is.na(x))/obs >= 0.8}) 
trainData <- trainData[,!nacols]
testData <- testData[,!nacols]
```

This leaves us with the following number of features in both data frames. We also verify that there are no missing values among remaining features.

```r
ncol(trainData)
```

```
## [1] 52
```

```r
ncol(testData)
```

```
## [1] 52
```

```r
sum(is.na(trainData))
```

```
## [1] 0
```

```r
sum(is.na(testData))
```

```
## [1] 0
```

We use **preProcess** function from **caret** to re-scale the data. 


```r
preProc <- preProcess(trainData, method = c("center", "scale"))
trainData <- predict(preProc, trainData)
testData <- predict(preProc, testData)
```
### Principal Component Analysis
We choose to apply PCA to further reduce the number of feature vectors and optimize the model. One can easily check that 37 principal components capture $99\%$ of the variance in our data set. We also apply $\log$ to smooth the data out. 

```r
mins1 <- apply(trainData, 2, function(x) min(x))
mins2 <- apply(testData, 2, function(x) min(x))
shift <- max(abs(min(mins1)),abs(min(mins2)))

preProcPCA <- preProcess(log10(trainData + shift + 1), method = "pca", pcaComp = 37) 
trainPC <- predict(preProcPCA, log10(trainData + shift + 1))
testPC <- predict(preProcPCA,log10(testData + shift + 1))
```

# Model training using Random Forests 
In **train** function we specify cross-validation with 6 folds as the preferred resampling method and pass the corresponding parameters to **trainControl**. We also specify the number of randomly-selected predictors at each split by passing the corresponding values to **tuneGrid**.   


```r
set.seed(2)
modFit <- train(trainPC, trainclass, method = "rf", trControl = trainControl(method = "cv", number = 6), allowParallel = T, tuneGrid = data.frame(.mtry = c(2,4,6)))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit
```

```
## Random Forest 
## 
## 7360 samples
##   37 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (6 fold) 
## 
## Summary of sample sizes: 6134, 6133, 6134, 6132, 6133, 6134, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.006        0.008   
##   4     0.9       0.9    0.009        0.01    
##   6     0.9       0.9    0.008        0.01    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 4.
```

In the following plots we can see the accuracy of our model vs. the nubmer of predictos. 


```r
trellis.par.set(caretTheme())
plot(modFit, main = "Accuracy.")
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13.png) 

As follows from above, the in-sample error is $\approx 4 \%$. 

## Cross-Validation and Out-of-Sample Error 
Now we can apply the model to our validation set and calculate the expected out-of-sample error. 


```r
pred <- predict(modFit, testPC)
```

Even though the data is higher-dimensional, we can observe the 5 clasters 
if we plot the projection of the data to the first two principal components. We can see a relatively small number of misclassified samples in the plot.  


```r
testPC$predRight <- pred == testclass
qplot(testPC[,1], testPC[,2], colour = predRight, data = testPC, main = "Test Data Predictors")
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15.png) 

At last, let's look at the confusion matrix and the accuracy of the model on the validation set.


```r
conf.matrix <- confusionMatrix(testclass, predict(modFit, testPC))
conf.matrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1370    8    3    5    9
##          B   30  900   18    0    1
##          C    3   23  815    7    7
##          D    7    0   45  748    4
##          E    7   11   18   25  840
## 
## Overall Statistics
##                                         
##                Accuracy : 0.953         
##                  95% CI : (0.947, 0.959)
##     No Information Rate : 0.289         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.94          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.967    0.955    0.907    0.953    0.976
## Specificity             0.993    0.988    0.990    0.986    0.985
## Pos Pred Value          0.982    0.948    0.953    0.930    0.932
## Neg Pred Value          0.987    0.989    0.979    0.991    0.995
## Prevalence              0.289    0.192    0.183    0.160    0.176
## Detection Rate          0.279    0.184    0.166    0.153    0.171
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.980    0.972    0.948    0.970    0.980
```
Thus, the expected out-of-sample error is $\approx 5 \%$.  

We conclude by demostrating how error rate changes with the number of trees in random forest.  


```r
model <- modFit$finalModel
model$confusion
```

```
##      A    B    C    D    E class.error
## A 2073    7    6    7    0    0.009556
## B   67 1321   29    4    3    0.072331
## C    4   40 1209   25    6    0.058411
## D    6    2   65 1126    7    0.066335
## E    3   18   24   23 1285    0.050259
```


```r
plot(model, main = "Error Rate Over Trees", log = "y")
legend("topright", legend=colnames(model$err.rate), col = c(1:6), pch=19)
```

![plot of chunk unnamed-chunk-18](figure/unnamed-chunk-18.png) 

