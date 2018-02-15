setwd('C:/Users/nija/Documents/titanic/')
# import libraries
library(ROCR)
library(caret)
library(glmnet)

# read in training set and interpret blank fields as NA 
train = read.csv("train.csv", na.strings = c("", "NA"), stringsAsFactors = F)

# read in test data
test = read.csv("test.csv", na.strings = c("", "NA"), stringsAsFactors = F)

# create complete data frame
fullData = as.data.frame(rbind(train[-2], test))

Y = as.matrix(train$Survived)

#################################################################################################### 
# exploring and cleaning the dataset 
#################################################################################################### 
isNA = sapply(fullData, function(x) sum(is.na(x)))

# cabin vector mostly empty - delete, lots of missing values in age - find title and set NA to the average of their title

# convert variables incorrectly coded as numeric to factor
fullData$Pclass = as.factor(fullData$Pclass)
fullData$Embarked = as.factor(fullData$Embarked)

# convert missing values in fare to the mean fare
fullData$Fare[is.na(fullData$Fare)] = median(fullData$Fare, na.rm = T) 

# convert missing values in embarked to the modal embarkment point
getmode = function(v) {
  uniqv = unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
} # function to find modal embarkment point

v = fullData$Embarked # embarkment col to be passed to getmode function
result = getmode(v) # returns the modal value in embarkment
fullData$Embarked[is.na(fullData$Embarked)] = result  # changes any NA in embarkment to modal embarkment point

#################################################################################################### 
# create new variable title based on name variable 
####################################################################################################
names = fullData$Name
title = gsub("^.*, (.*?)\\..*$", "\\1", names)
fullData$title = title
#print(table(title))

# condense female titles into Miss or Mrs 
fullData$title[fullData$title == 'Mlle'] = 'Miss' 
fullData$title[fullData$title == 'Ms'] = 'Miss'
fullData$title[fullData$title == 'Mme'] = 'Mrs' 
fullData$title[fullData$title == 'Lady'] = 'Miss'
fullData$title[fullData$title == 'Dona'] = 'Miss'

# condense officials into one group
fullData$title[fullData$title == 'Capt'] = 'Officer' 
fullData$title[fullData$title == 'Col'] = 'Officer' 
fullData$title[fullData$title == 'Major'] = 'Officer'
fullData$title[fullData$title == 'Dr'] = 'Officer'
fullData$title[fullData$title == 'Rev'] = 'Officer'
fullData$title[fullData$title == 'Don'] = 'Officer'
fullData$title[fullData$title == 'Sir'] = 'Officer'
fullData$title[fullData$title == 'the Countess'] = 'Officer'
fullData$title[fullData$title == 'Jonkheer'] = 'Officer'

fullData$title = as.factor(fullData$title) # change to title to factor

#################################################################################################### 
# find missing values in age and convert to the mean age according to their title 
####################################################################################################
# find mean age of each title
whichMaster = which(fullData$title == 'Master') # find rows that contain master
masterAges = fullData$Age[whichMaster] # find the masters ages
masterMean = mean(masterAges, na.rm = T) # calculate mean
ms = which(is.na(fullData$Age) == T & fullData$title == 'Master', arr.ind = T) # get row numbers from whichMaster that have ages that are missing 
fullData$Age[ms] = masterMean

whichMiss = which(fullData$title == 'Miss') 
missAges = fullData$Age[whichMiss]
missMean = mean(missAges, na.rm = T)
miss  = which(is.na(fullData$Age) == T & fullData$title == 'Miss', arr.ind = T)
fullData$Age[miss] = missMean

whichMr = which(fullData$title == 'Mr') 
mrAges = fullData$Age[whichMr]
mrMean = mean(missAges, na.rm = T)
mr = which(is.na(fullData$Age) == T & fullData$title == 'Mr', arr.ind = T)
fullData$Age[mr] = mrMean

whichMrs = which(fullData$title == 'Mrs') 
mrsAges = fullData$Age[whichMrs]
mrsMean = mean(mrsAges, na.rm = T)
mrs = which(is.na(fullData$Age) == T & fullData$title == 'Mrs', arr.ind = T)
fullData$Age[mrs] = mrsMean

whichOfficer = which(fullData$title == 'Officer') 
officerAges = fullData$Age[whichOfficer]
officerMean = mean(officerAges, na.rm = T)
off = which(is.na(fullData$Age) == T & fullData$title == 'Officer', arr.ind = T)
fullData$Age[off] = officerMean

####################################################################################################
# create new variable family size 
####################################################################################################
fullData$FamilySize =fullData$SibSp + fullData$Parch + 1 

fullData$FamilySized[fullData$FamilySize == 1]   = 'Single'
fullData$FamilySized[fullData$FamilySize < 5 & fullData$FamilySize >= 2]   = 'Small'
fullData$FamilySized[fullData$FamilySize >= 5]   = 'Big'

fullData$FamilySized = as.factor(fullData$FamilySized)

####################################################################################################
# full training data frame 
####################################################################################################
input = fullData[1:891, c("Pclass", "FamilySized", "title", "Sex", "Age")]
input = cbind(input, Y)

####################################################################################################
# find coefficients using lasso regression
####################################################################################################
x = model.matrix(Y ~ ., input)
model = cv.glmnet(x, Y, alpha = 0, family = "binomial", type.measure = "class")
lambdaMin = model$lambda.min
lambda1se = model$lambda.1se
coef(model, s = lambda1se)

predTrain = predict(model, x, type = "class")
lassoPred = ifelse(predTrain > 0.5, 1, 0) 

# asessing model accuracy with ROCR library to determine cutoff point for predicted values
ROCRpred = prediction(lassoPred, input$Y) 
ROCRperf = performance(ROCRpred, 'acc')
ind = which.max( slot(ROCRperf, "y.values")[[1]] ) # find which cutpoint has the highest accuracy in ROCRperf
cutoff = slot(ROCRperf, "x.values")[[1]][ind] # what is the cutpoint that corresponds to that level of accuracy

testInput = fullData[892:1309, c("Pclass", "FamilySized", "title", "Sex", "Age")] # test data frame
XX = model.matrix(~., testInput)
lassoProb = predict(model, XX, type = "response")
lassoPred = ifelse(lassoProb > cutoff, 1, 0) 

survived = cbind.data.frame(test$PassengerId, lassoPred)
names(survived)[1] = "PassengerID"
names(survived)[2] = "Survived"
write.csv(survived, "Survived.csv", row.names = F)
print(summary(model))
