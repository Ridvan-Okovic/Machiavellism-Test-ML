#setwd("C:/Users/mirza/Desktop/ML")

data <- read.csv("data.csv", stringsAsFactors = FALSE)

##----------------------------- PREPROCESS START -----------------------------##

# Preprocessing - selecting columns that are going to be used,
# Remove non-existant fields and missing values

# import libraries 
library(mlbench)
library(caret)
library(gmodels)
library(class)
library(dplyr)
library(randomForest)
library(MLeval)
library(pROC)
library("ROCR")
library(MLeval)

# Choose 15% of the data to use for training and test 
# We are choosing randomly so the model would not be biased
dataPercent <- sort(sample(nrow(data), nrow(data)*.50))
# Now data_selection stores 15% of actual data that we will use
data <- data[dataPercent,]

# select columns from the data set on which we will be used
data <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, Q9A, 
               Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
               Q19A, Q20A, TIPI1, TIPI2, TIPI3, TIPI4, TIPI5, TIPI6, 
               TIPI7, TIPI8, TIPI9, TIPI10)

histogram(data$TIPI1)

# Convert all values from 0 to NA so we can remove them
# Questions 1 through 20
data[!is.na(data$Q1A) & 
       data$Q1A == 0, ] <- NA
data[!is.na(data$Q2A) & 
       data$Q2A == 0, ] <- NA
data[!is.na(data$Q3A) & 
       data$Q3A == 0, ] <- NA
data[!is.na(data$Q4A) & 
       data$Q4A == 0, ] <- NA
data[!is.na(data$Q5A) & 
       data$Q5A == 0, ] <- NA
data[!is.na(data$Q6A) & 
       data$Q6A == 0, ] <- NA
data[!is.na(data$Q7A) & 
       data$Q7A == 0, ] <- NA
data[!is.na(data$Q8A) & 
       data$Q8A == 0, ] <- NA
data[!is.na(data$Q9A) & 
       data$Q9A == 0, ] <- NA
data[!is.na(data$Q10A) & 
       data$Q10A == 0, ] <- NA
data[!is.na(data$Q11A) & 
       data$Q11A == 0, ] <- NA
data[!is.na(data$Q12A) & 
       data$Q12A == 0, ] <- NA
data[!is.na(data$Q13A) & 
       data$Q13A == 0, ] <- NA
data[!is.na(data$Q14A) & 
       data$Q14A == 0, ] <- NA
data[!is.na(data$Q15A) & 
       data$Q15A == 0, ] <- NA
data[!is.na(data$Q16A) & 
       data$Q16A == 0, ] <- NA
data[!is.na(data$Q17A) & 
       data$Q17A == 0, ] <- NA
data[!is.na(data$Q18A) & 
       data$Q18A == 0, ] <- NA
data[!is.na(data$Q19A) & 
       data$Q19A == 0, ] <- NA
data[!is.na(data$Q20A) & 
       data$Q20A == 0, ] <- NA

# Personality Ten Item Traits removal of non-existant fields
# Since less than 8% of surveyees picked neither agree or disagree we will 
# remove the class to make it easier for the model
data[!is.na(data$TIPI1) & 
       data$TIPI1 == 0, ] <- NA
data[!is.na(data$TIPI1) & 
       data$TIPI1 == 4, ] <- NA

data[!is.na(data$TIPI2) & 
       data$TIPI2 == 0, ] <- NA
data[!is.na(data$TIPI2) & 
       data$TIPI2 == 4, ] <- NA

data[!is.na(data$TIPI3) & 
       data$TIPI3 == 0, ] <- NA
data[!is.na(data$TIPI3) & 
       data$TIPI3 == 4, ] <- NA

data[!is.na(data$TIPI4) & 
       data$TIPI4 == 0, ] <- NA
data[!is.na(data$TIPI4) & 
       data$TIPI4 == 4, ] <- NA

data[!is.na(data$TIPI5) & 
       data$TIPI5 == 0, ] <- NA
data[!is.na(data$TIPI5) & 
       data$TIPI5 == 4, ] <- NA

data[!is.na(data$TIPI6) & 
       data$TIPI6 == 0, ] <- NA
data[!is.na(data$TIPI6) & 
       data$TIPI6 == 4, ] <- NA

data[!is.na(data$TIPI7) & 
       data$TIPI7 == 0, ] <- NA
data[!is.na(data$TIPI7) & 
       data$TIPI7 == 4, ] <- NA

data[!is.na(data$TIPI8) & 
       data$TIPI8 == 0, ] <- NA
data[!is.na(data$TIPI8) & 
       data$TIPI8 == 4, ] <- NA

data[!is.na(data$TIPI9) & 
       data$TIPI9 == 0, ] <- NA
data[!is.na(data$TIPI9) & 
       data$TIPI9 == 4, ] <- NA

data[!is.na(data$TIPI10) & 
       data$TIPI10 == 0, ] <- NA
data[!is.na(data$TIPI10) & 
       data$TIPI10 == 4, ] <- NA

# Clean up and remove NA values
data <- data[complete.cases(data), ]


# Store those who disagree strongly, moderately and a little as disagree,
# otherwise agree, since most of the surveyees when picking disagree pick the 
# option of strongly or moderately disagree, so again we are helping the model

# TIPI_1
data[data[, "TIPI1"] == 1, "TIPI1"] <- 1
data[data[, "TIPI1"] == 2, "TIPI1"] <- 1
data[data[, "TIPI1"] == 3, "TIPI1"] <- 1

data[data[, "TIPI1"] == 5, "TIPI1"] <- 2
data[data[, "TIPI1"] == 6, "TIPI1"] <- 2
data[data[, "TIPI1"] == 7, "TIPI1"] <- 2


# TIPI_2
data[data[, "TIPI2"] == 1, "TIPI2"] <- 1
data[data[, "TIPI2"] == 2, "TIPI2"] <- 1
data[data[, "TIPI2"] == 3, "TIPI2"] <- 1

data[data[, "TIPI2"] == 5, "TIPI2"] <- 2
data[data[, "TIPI2"] == 6, "TIPI2"] <- 2
data[data[, "TIPI2"] == 7, "TIPI2"] <- 2


data[data[, "TIPI3"] == 1, "TIPI3"] <- 1
data[data[, "TIPI3"] == 2, "TIPI3"] <- 1
data[data[, "TIPI3"] == 3, "TIPI3"] <- 1

data[data[, "TIPI3"] == 5, "TIPI3"] <- 2
data[data[, "TIPI3"] == 6, "TIPI3"] <- 2
data[data[, "TIPI3"] == 7, "TIPI3"] <- 2


data[data[, "TIPI4"] == 1, "TIPI4"] <- 1
data[data[, "TIPI4"] == 2, "TIPI4"] <- 1
data[data[, "TIPI4"] == 3, "TIPI4"] <- 1

data[data[, "TIPI4"] == 5, "TIPI4"] <- 2
data[data[, "TIPI4"] == 6, "TIPI4"] <- 2
data[data[, "TIPI4"] == 7, "TIPI4"] <- 2


data[data[, "TIPI5"] == 1, "TIPI5"] <- 1
data[data[, "TIPI5"] == 2, "TIPI5"] <- 1
data[data[, "TIPI5"] == 3, "TIPI5"] <- 1

data[data[, "TIPI5"] == 5, "TIPI5"] <- 2
data[data[, "TIPI5"] == 6, "TIPI5"] <- 2
data[data[, "TIPI5"] == 7, "TIPI5"] <- 2


data[data[, "TIPI6"] == 1, "TIPI6"] <- 1
data[data[, "TIPI6"] == 2, "TIPI6"] <- 1
data[data[, "TIPI6"] == 3, "TIPI6"] <- 1

data[data[, "TIPI6"] == 5, "TIPI6"] <- 2
data[data[, "TIPI6"] == 6, "TIPI6"] <- 2
data[data[, "TIPI6"] == 7, "TIPI6"] <- 2


data[data[, "TIPI7"] == 1, "TIPI7"] <- 1
data[data[, "TIPI7"] == 2, "TIPI7"] <- 1
data[data[, "TIPI7"] == 3, "TIPI7"] <- 1

data[data[, "TIPI7"] == 5, "TIPI7"] <- 2
data[data[, "TIPI7"] == 6, "TIPI7"] <- 2
data[data[, "TIPI7"] == 7, "TIPI7"] <- 2


data[data[, "TIPI8"] == 1, "TIPI8"] <- 1
data[data[, "TIPI8"] == 2, "TIPI8"] <- 1
data[data[, "TIPI8"] == 3, "TIPI8"] <- 1

data[data[, "TIPI8"] == 5, "TIPI8"] <- 2
data[data[, "TIPI8"] == 6, "TIPI8"] <- 2
data[data[, "TIPI8"] == 7, "TIPI8"] <- 2


data[data[, "TIPI9"] == 1, "TIPI9"] <- 1
data[data[, "TIPI9"] == 2, "TIPI9"] <- 1
data[data[, "TIPI9"] == 3, "TIPI9"] <- 1

data[data[, "TIPI9"] == 5, "TIPI9"] <- 2
data[data[, "TIPI9"] == 6, "TIPI9"] <- 2
data[data[, "TIPI9"] == 7, "TIPI9"] <- 2


data[data[, "TIPI10"] == 1, "TIPI10"] <- 1
data[data[, "TIPI10"] == 2, "TIPI10"] <- 1
data[data[, "TIPI10"] == 3, "TIPI10"] <- 1

data[data[, "TIPI10"] == 5, "TIPI10"] <- 2
data[data[, "TIPI10"] == 6, "TIPI10"] <- 2
data[data[, "TIPI10"] == 7, "TIPI10"] <- 2


## -------------------------- FAETURE SELECTION ---------------------------- ##
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cross validation
                      number = 10) # number of folds

rfeSolution <- rfe(x = data[, 1:20], y = data[, 21], 
                   sizes = c(1:20), rfeControl = control)
rfeSolution
ggplot(data = rfeSolution, metric = "Accuracy") + theme_bw()

varimp_data <- data.frame(feature = row.names(varImp(rfeSolution))[1:20],
                          importance = varImp(rfeSolution)[1:20, 1])

ggplot(data = varimp_data, aes(x = reorder(feature, -importance), y = importance, 
                                fill = feature)) + geom_bar(stat="identity") + 
                                labs(x = "Features", y = "Variable Importance") + 
                                geom_text(aes(label = round(importance, 2)), 
                                vjust=1.6, color="white", size=4) + theme_bw() + 
                                theme(legend.position = "none")

# Correlation matrix suggested that we remove questions 6 and 9
correlationMatrix <- cor(data)
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
print(highlyCorrelated)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(data$TIPI1~., data=data, method="lvq", preProcess="scale", trControl=control)
## -------------------------------------------------------------------------- ##

# Create 10 new data sets which will have only one output column TIPI1 - TIPI10
# And for each TIPI data set create train and test sets
train_test_percent <- sort(sample(nrow(data), nrow(data)*.7))

data_TIPI1 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, Q9A, Q10A, 
                     Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI1)
TIPI1_Train <- data_TIPI1[train_test_percent,]
TIPI1_Test <- data_TIPI1[-train_test_percent,]


data_TIPI2 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI2)
TIPI2_Train <- data_TIPI2[train_test_percent,]
TIPI2_Test <- data_TIPI2[-train_test_percent,]

data_TIPI3 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI3)
TIPI3_Train <- data_TIPI3[train_test_percent,]
TIPI3_Test <- data_TIPI3[-train_test_percent,]

data_TIPI4 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI4)
TIPI4_Train <- data_TIPI4[train_test_percent,]
TIPI4_Test <- data_TIPI4[-train_test_percent,]

data_TIPI5 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI5)
TIPI5_Train <- data_TIPI5[train_test_percent,]
TIPI5_Test <- data_TIPI5[-train_test_percent,]

data_TIPI6 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI6)
TIPI6_Train <- data_TIPI6[train_test_percent,]
TIPI6_Test <- data_TIPI6[-train_test_percent,]

data_TIPI7 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI7)
TIPI7_Train <- data_TIPI7[train_test_percent,]
TIPI7_Test <- data_TIPI7[-train_test_percent,]

data_TIPI8 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI8)
TIPI8_Train <- data_TIPI8[train_test_percent,]
TIPI8_Test <- data_TIPI8[-train_test_percent,]

data_TIPI9 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI9)
TIPI9_Train <- data_TIPI9[train_test_percent,]
TIPI9_Test <- data_TIPI9[-train_test_percent,]

data_TIPI10 <- select(data, Q1A, Q2A, Q3A, Q4A, Q5A, Q6A, Q7A, Q8A, 
                     Q9A, Q10A, Q11A, Q12A, Q13A, Q14A, Q15A, Q16A, Q17A, Q18A, 
                     Q19A, Q20A, TIPI10)
TIPI10_Train <- data_TIPI10[train_test_percent,]
TIPI10_Test <- data_TIPI10[-train_test_percent,]

## ----------------------------- PREPROCESS END ----------------------------- ##


## -------------------------------- KNN START ------------------------------- ##

# Data from Q1 to Q20 represents the questions the respondents were asked,
# TIPI1 to TIPI10 represent "Ten Item Personality Inventory",
# The TIPI items were rated "I see myself as:"______ such that: 
# 1 - Disagree  and 2 - Agree 


# kNN model for first personality trait

TIPI1_kNN <- knn(train=TIPI1_Train, test=TIPI1_Test, TIPI1_Train$TIPI1, k=70, prob = TRUE)
TIPI1_CF <- confusionMatrix(data=TIPI1_kNN, reference=as.factor(TIPI1_Test$TIPI1))
TIPI1_CF
roc(TIPI1_Test$TIPI1, attributes(TIPI1_kNN)$prob)

# kNN model for second personality trait

TIPI2_kNN <- knn(train=TIPI2_Train, test=TIPI2_Test, as.factor(TIPI2_Train$TIPI2), k=70, prob=TRUE)
TIPI2_CF <- confusionMatrix(data=TIPI2_kNN, reference=as.factor(TIPI2_Test$TIPI2))
TIPI2_CF
roc(TIPI2_Test$TIPI2, attributes(TIPI2_kNN)$prob)

# kNN model for third personality trait

TIPI3_kNN <- knn(train=TIPI3_Train, test=TIPI3_Test, as.factor(TIPI3_Train$TIPI3), k=70, prob=TRUE)
TIPI3_CF <- confusionMatrix(data=TIPI3_kNN, reference=as.factor(TIPI3_Test$TIPI3))
TIPI3_CF
roc(TIPI3_Test$TIPI3, attributes(TIPI3_kNN)$prob)

# kNN model for fourth personality trait

TIPI4_kNN <- knn(train=TIPI4_Train, test=TIPI4_Test, as.factor(TIPI4_Train$TIPI4), k=70, prob=TRUE)
TIPI4_CF <- confusionMatrix(data=TIPI4_kNN, reference=as.factor(TIPI4_Test$TIPI4))
TIPI4_CF
roc(TIPI4_Test$TIPI4, attributes(TIPI4_kNN)$prob)

# kNN model for fifth personality trait

TIPI5_kNN <- knn(train=TIPI5_Train, test=TIPI5_Test, as.factor(TIPI5_Train$TIPI5), k=70, prob=TRUE)
TIPI5_CF <- confusionMatrix(data=TIPI5_kNN, reference=as.factor(TIPI5_Test$TIPI5))
TIPI5_CF
roc(TIPI5_Test$TIPI5, attributes(TIPI5_kNN)$prob)

# kNN model for sixth personality trait

TIPI6_kNN <- knn(train=TIPI6_Train, test=TIPI2_Test, as.factor(TIPI6_Train$TIPI6), k=70, prob=TRUE)
TIPI6_CF <- confusionMatrix(data=TIPI6_kNN, reference=as.factor(TIPI6_Test$TIPI6))
TIPI6_CF
roc(TIPI6_Test$TIPI6, attributes(TIPI6_kNN)$prob)

# kNN model for seventh personality trait

TIPI7_kNN <- knn(train=TIPI7_Train, test=TIPI7_Test, as.factor(TIPI7_Train$TIPI7), k=70, prob=TRUE)
TIPI7_CF <- confusionMatrix(data=TIPI7_kNN, reference=as.factor(TIPI7_Test$TIPI7))
TIPI7_CF
roc(TIPI7_Test$TIPI7, attributes(TIPI7_kNN)$prob)

# kNN model for eight personality trait

TIPI8_kNN <- knn(train=TIPI8_Train, test=TIPI8_Test, as.factor(TIPI8_Train$TIPI8), k=70, prob=TRUE)
TIPI8_CF <- confusionMatrix(data=TIPI8_kNN, reference=as.factor(TIPI8_Test$TIPI8))
TIPI8_CF
roc(TIPI8_Test$TIPI8, attributes(TIPI8_kNN)$prob)

# kNN model for ninth personality trait

TIPI9_kNN <- knn(train=TIPI9_Train, test=TIPI9_Test, as.factor(TIPI9_Train$TIPI9), k=70, prob=TRUE)
TIPI9_CF <- confusionMatrix(data=TIPI9_kNN, reference=as.factor(TIPI9_Test$TIPI9))
TIPI9_CF
roc(TIPI9_Test$TIPI9, attributes(TIPI9_kNN)$prob)

# kNN model for tenth personality trait

TIPI10_kNN <- knn(train=TIPI10_Train, test=TIPI10_Test, as.factor(TIPI10_Train$TIPI10), k=70, prob=TRUE)
TIPI10_CF <- confusionMatrix(data=TIPI10_kNN, reference=as.factor(TIPI10_Test$TIPI10))
TIPI10_CF
roc(TIPI10_Test$TIPI10, attributes(TIPI10_kNN)$prob)


TIPI1_Train[TIPI1_Train$TIPI1 == 1, ]$TIPI1 <- "Disagree"
TIPI1_Train[TIPI1_Train$TIPI1 == 2, ]$TIPI1 <- "Agree"
TIPI1_Train$TIPI1 <- as.factor(TIPI1_Train$TIPI1)
TIPI1_Test[TIPI1_Test$TIPI1 == 1, ]$TIPI1 <- "Disagree"
TIPI1_Test[TIPI1_Test$TIPI1 == 2, ]$TIPI1 <- "Agree"
TIPI1_Test$TIPI1 <- as.factor(TIPI1_Test$TIPI1)


train_control <- trainControl(method = "cv", number = 10, 
                              summaryFunction = twoClassSummary, 
                              classProbs = TRUE, savePredictions = TRUE)
TIPI1_Kfold <- train(TIPI1 ~., data = TIPI1_Train, method = "knn",
                     tuneGrid = expand.grid(k = 20:70),
                     trControl = train_control, metric = "ROC")
print(TIPI1_Kfold$results)
TIPI1_kNN_predict <- predict(TIPI1_Kfold, newdata = TIPI1_Test)
confusionMatrix(TIPI1_kNN_predict, TIPI1_Test$TIPI1)

# for accuracy
train_control_acc <- trainControl(method = "cv", number = 10)
TIPI1_Kfold_Acc <- train(TIPI1 ~., data = TIPI1_Train, method = "knn",
                         tuneGrid = expand.grid(k = 20:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI1_Kfold_Acc$results)
display1 <- evalm(TIPI1_Kfold, gnames="knn")

TIPI2_Train[TIPI2_Train$TIPI2 == 1, ]$TIPI2 <- "Disagree"
TIPI2_Train[TIPI2_Train$TIPI2 == 2, ]$TIPI2 <- "Agree"
TIPI2_Train$TIPI2 <- as.factor(TIPI2_Train$TIPI2)
TIPI2_Test[TIPI2_Test$TIPI2 == 1, ]$TIPI2 <- "Disagree"
TIPI2_Test[TIPI2_Test$TIPI2 == 2, ]$TIPI2 <- "Agree"
TIPI2_Test$TIPI2 <- as.factor(TIPI2_Test$TIPI2)


TIPI2_Kfold <- train(TIPI2 ~., data = TIPI2_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI2_Kfold$results)
TIPI2_kNN_predict <- predict(TIPI2_Kfold, newdata = TIPI2_Test)
confusionMatrix(TIPI2_kNN_predict, TIPI2_Test$TIPI2)
display1 <- evalm(TIPI2_Kfold, gnames="knn")
# for accuracy
TIPI2_Kfold_Acc <- train(TIPI2 ~., data = TIPI2_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI2_Kfold_Acc$results)

TIPI3_Train[TIPI3_Train$TIPI3 == 1, ]$TIPI3 <- "Disagree"
TIPI3_Train[TIPI3_Train$TIPI3 == 2, ]$TIPI3 <- "Agree"
TIPI3_Train$TIPI3 <- as.factor(TIPI3_Train$TIPI3)
TIPI3_Test[TIPI3_Test$TIPI3 == 1, ]$TIPI3 <- "Disagree"
TIPI3_Test[TIPI3_Test$TIPI3 == 2, ]$TIPI3 <- "Agree"
TIPI3_Test$TIPI3 <- as.factor(TIPI3_Test$TIPI3)

TIPI3_Kfold <- train(TIPI3 ~., data = TIPI3_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI3_Kfold$results)
# for accuracy
TIPI3_kNN_predict <- predict(TIPI3_Kfold, newdata = TIPI3_Test)
confusionMatrix(TIPI3_kNN_predict, TIPI3_Test$TIPI3)
display3 <- evalm(TIPI3_Kfold, gnames="knn")
TIPI3_Kfold_Acc <- train(TIPI3 ~., data = TIPI3_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI3_Kfold_Acc$results)

TIPI4_Train[TIPI4_Train$TIPI4 == 1, ]$TIPI4 <- "Disagree"
TIPI4_Train[TIPI4_Train$TIPI4 == 2, ]$TIPI4 <- "Agree"
TIPI4_Train$TIPI4 <- as.factor(TIPI4_Train$TIPI4)
TIPI4_Test[TIPI4_Test$TIPI4 == 1, ]$TIPI4 <- "Disagree"
TIPI4_Test[TIPI4_Test$TIPI4 == 2, ]$TIPI4 <- "Agree"
TIPI4_Test$TIPI4 <- as.factor(TIPI4_Test$TIPI4)

TIPI4_Kfold <- train(TIPI4 ~., data = TIPI4_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI4_Kfold$results)
TIPI4_kNN_predict <- predict(TIPI4_Kfold, newdata = TIPI4_Test)
confusionMatrix(TIPI4_kNN_predict, TIPI4_Test$TIPI4)
display4 <- evalm(TIPI4_Kfold, gnames="knn")
# for accuracy

TIPI4_Kfold_Acc <- train(TIPI4 ~., data = TIPI4_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI4_Kfold_Acc$results)

TIPI5_Train[TIPI5_Train$TIPI5 == 1, ]$TIPI5 <- "Disagree"
TIPI5_Train[TIPI5_Train$TIPI5 == 2, ]$TIPI5 <- "Agree"
TIPI5_Train$TIPI5 <- as.factor(TIPI5_Train$TIPI5)
TIPI5_Test[TIPI5_Test$TIPI5 == 1, ]$TIPI5 <- "Disagree"
TIPI5_Test[TIPI5_Test$TIPI5 == 2, ]$TIPI5 <- "Agree"
TIPI5_Test$TIPI5 <- as.factor(TIPI5_Test$TIPI5)

TIPI5_Kfold <- train(TIPI5 ~., data = TIPI5_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI5_Kfold$results)
TIPI5_kNN_predict <- predict(TIPI5_Kfold, newdata = TIPI5_Test)
confusionMatrix(TIPI5_kNN_predict, TIPI5_Test$TIPI5)
display5 <- evalm(TIPI5_Kfold, gnames="knn")
# for accuracy

TIPI5_Kfold_Acc <- train(TIPI5 ~., data = TIPI5_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI5_Kfold_Acc$results)

TIPI6_Train[TIPI6_Train$TIPI6 == 1, ]$TIPI6 <- "Disagree"
TIPI6_Train[TIPI6_Train$TIPI6 == 2, ]$TIPI6 <- "Agree"
TIPI6_Train$TIPI6 <- as.factor(TIPI6_Train$TIPI6)
TIPI6_Test[TIPI6_Test$TIPI6 == 1, ]$TIPI6 <- "Disagree"
TIPI6_Test[TIPI6_Test$TIPI6 == 2, ]$TIPI6 <- "Agree"
TIPI6_Test$TIPI6 <- as.factor(TIPI6_Test$TIPI6)


TIPI6_Kfold <- train(TIPI6 ~., data = TIPI6_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI6_Kfold$results)
TIPI6_kNN_predict <- predict(TIPI6_Kfold, newdata = TIPI6_Test)
confusionMatrix(TIPI6_kNN_predict, TIPI6_Test$TIPI6)
display6 <- evalm(TIPI6_Kfold, gnames="knn")
# for accuracy

TIPI6_Kfold_Acc <- train(TIPI6 ~., data = TIPI6_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI6_Kfold_Acc$results)

TIPI7_Train[TIPI7_Train$TIPI7 == 1, ]$TIPI7 <- "Disagree"
TIPI7_Train[TIPI7_Train$TIPI7 == 2, ]$TIPI7 <- "Agree"
TIPI7_Train$TIPI7 <- as.factor(TIPI7_Train$TIPI7)
TIPI7_Test[TIPI7_Test$TIPI7 == 1, ]$TIPI7 <- "Disagree"
TIPI7_Test[TIPI7_Test$TIPI7 == 2, ]$TIPI7 <- "Agree"
TIPI7_Test$TIPI7 <- as.factor(TIPI7_Test$TIPI7)


TIPI7_Kfold <- train(TIPI7 ~., data = TIPI7_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI7_Kfold$results)
TIPI7_kNN_predict <- predict(TIPI7_Kfold, newdata = TIPI7_Test)
confusionMatrix(TIPI7_kNN_predict, TIPI7_Test$TIPI7)
display7 <- evalm(TIPI7_Kfold, gnames="knn")
# for accuracy

TIPI7_Kfold_Acc <- train(TIPI7 ~., data = TIPI7_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI7_Kfold_Acc$results)

TIPI8_Train[TIPI8_Train$TIPI8 == 1, ]$TIPI8 <- "Disagree"
TIPI8_Train[TIPI8_Train$TIPI8 == 2, ]$TIPI8 <- "Agree"
TIPI8_Train$TIPI8 <- as.factor(TIPI8_Train$TIPI8)
TIPI8_Test[TIPI8_Test$TIPI8 == 1, ]$TIPI8 <- "Disagree"
TIPI8_Test[TIPI8_Test$TIPI8 == 2, ]$TIPI8 <- "Agree"
TIPI8_Test$TIPI8 <- as.factor(TIPI8_Test$TIPI8)


TIPI8_Kfold <- train(TIPI8 ~., data = TIPI8_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI8_Kfold$results)
TIPI8_kNN_predict <- predict(TIPI8_Kfold, newdata = TIPI8_Test)
confusionMatrix(TIPI8_kNN_predict, TIPI8_Test$TIPI8)
display8 <- evalm(TIPI8_Kfold, gnames="knn")
# for accuracy

TIPI8_Kfold_Acc <- train(TIPI8 ~., data = TIPI8_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI8_Kfold_Acc$results)

TIPI9_Train[TIPI9_Train$TIPI9 == 1, ]$TIPI9 <- "Disagree"
TIPI9_Train[TIPI9_Train$TIPI9 == 2, ]$TIPI9 <- "Agree"
TIPI9_Train$TIPI9 <- as.factor(TIPI9_Train$TIPI9)
TIPI9_Test[TIPI9_Test$TIPI9 == 1, ]$TIPI9 <- "Disagree"
TIPI9_Test[TIPI9_Test$TIPI9 == 2, ]$TIPI9 <- "Agree"
TIPI9_Test$TIPI9 <- as.factor(TIPI9_Test$TIPI9)

TIPI9_Kfold <- train(TIPI9 ~., data = TIPI9_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI9_Kfold$results)
TIPI9_kNN_predict <- predict(TIPI9_Kfold, newdata = TIPI9_Test)
confusionMatrix(TIPI9_kNN_predict, TIPI9_Test$TIPI9)
display9 <- evalm(TIPI9_Kfold, gnames="knn")
# for accuracy
TIPI9_Kfold_Acc <- train(TIPI9 ~., data = TIPI9_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI9_Kfold_Acc$results)

TIPI10_Train[TIPI10_Train$TIPI10 == 1, ]$TIPI10 <- "Disagree"
TIPI10_Train[TIPI10_Train$TIPI10 == 2, ]$TIPI10 <- "Agree"
TIPI10_Train$TIPI10 <- as.factor(TIPI10_Train$TIPI10)
TIPI10_Test[TIPI10_Test$TIPI10 == 1, ]$TIPI10 <- "Disagree"
TIPI10_Test[TIPI10_Test$TIPI10 == 2, ]$TIPI10 <- "Agree"
TIPI10_Test$TIPI10 <- as.factor(TIPI10_Test$TIPI10)

TIPI10_Kfold <- train(TIPI10 ~., data = TIPI10_Train, method = "knn",
                     tuneGrid = expand.grid(k = 50:70),
                     trControl = train_control, metric = "ROC")
print(TIPI10_Kfold$results)
TIPI10_kNN_predict <- predict(TIPI10_Kfold, newdata = TIPI10_Test)
confusionMatrix(TIPI10_kNN_predict, TIPI10_Test$TIPI10)
display10 <- evalm(TIPI10_Kfold, gnames="knn")
# for accuracy
TIPI10_Kfold_Acc <- train(TIPI10 ~., data = TIPI10_Train, method = "knn",
                         tuneGrid = expand.grid(k = 50:70),
                         trControl = train_control_acc, metric="Accuracy")
print(TIPI10_Kfold_Acc$results)


## --------------------------------- KNN END -------------------------------- ##


## --------------------------- RANDOM FOREST START -------------------------- ##
library(randomForest)

TIPI1_Train$TIPI1 <- as.factor(TIPI1_Train$TIPI1)
TIPI1_Test$TIPI1 <- as.factor(TIPI1_Test$TIPI1)

TIPI2_Train$TIPI2 <- as.factor(TIPI2_Train$TIPI2)
TIPI2_Test$TIPI2 <- as.factor(TIPI2_Test$TIPI2)

TIPI3_Train$TIPI3 <- as.factor(TIPI3_Train$TIPI3)
TIPI3_Test$TIPI3 <- as.factor(TIPI3_Test$TIPI3)

TIPI4_Train$TIPI4 <- as.factor(TIPI4_Train$TIPI4)
TIPI4_Test$TIPI4 <- as.factor(TIPI4_Test$TIPI4)

TIPI5_Train$TIPI5 <- as.factor(TIPI5_Train$TIPI5)
TIPI5_Test$TIPI5 <- as.factor(TIPI5_Test$TIPI5)

TIPI6_Train$TIPI6 <- as.factor(TIPI6_Train$TIPI6)
TIPI6_Test$TIPI6 <- as.factor(TIPI6_Test$TIPI6)

TIPI7_Train$TIPI7 <- as.factor(TIPI7_Train$TIPI7)
TIPI7_Test$TIPI7 <- as.factor(TIPI7_Test$TIPI7)

TIPI8_Train$TIPI8 <- as.factor(TIPI8_Train$TIPI8)
TIPI8_Test$TIPI8 <- as.factor(TIPI8_Test$TIPI8)

TIPI9_Train$TIPI9 <- as.factor(TIPI9_Train$TIPI9)
TIPI9_Test$TIPI9 <- as.factor(TIPI9_Test$TIPI9)

TIPI10_Train$TIPI10 <- as.factor(TIPI10_Train$TIPI10)
TIPI10_Test$TIPI10 <- as.factor(TIPI10_Test$TIPI10)





TIPI1_RandomForest <- randomForest(TIPI1_Train$TIPI1 ~ ., data=TIPI1_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI1_RandomForest

rfPredTIPI1 <- predict(TIPI1_RandomForest, newdata = TIPI1_Test, type = "prob")[,2]
rfPredTIPI1 <- prediction(rfPred, labels = TIPI1_Test$TIPI1)
rf_TIPI1_auc <- performance(rfPred, measure = "auc")
rf_TIPI1_auc_value <- rf_TIPI1_auc@y.values
rf_TIPI1_auc_value

TIPI1_CF_RF <- confusionMatrix(data = TIPI1_Test$TIPI1, reference=TIPI1_Pred)
TIPI1_CF_RF

TIPI2_RandomForest <- randomForest(TIPI2_Train$TIPI2 ~ ., data=TIPI2_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI2_RandomForest

rfPredTIPI2 <- predict(TIPI2_RandomForest, newdata = TIPI2_Test, type = "prob")[,2]
rfPredTIPI2 <- prediction(rfPredTIPI2, labels = TIPI2_Test$TIPI2)
rf_TIPI2_auc <- performance(rfPredTIPI2, measure = "auc")
rf_TIPI2_auc_value <- rf_TIPI2_auc@y.values
rf_TIPI2_auc_value

TIPI2_CF_RF <- confusionMatrix(data = TIPI2_Test$TIPI2, reference=TIPI2_Pred)
TIPI2_CF_RF

TIPI3_RandomForest <- randomForest(TIPI3_Train$TIPI3 ~ ., data=TIPI3_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI3_RandomForest

rfPredTIPI3 <- predict(TIPI3_RandomForest, newdata = TIPI3_Test, type = "prob")[,2]
rfPredTIPI3 <- prediction(rfPredTIPI3, labels = TIPI3_Test$TIPI3)
rf_TIPI3_auc <- performance(rfPredTIPI3, measure = "auc")
rf_TIPI3_auc_value <- rf_TIPI3_auc@y.values
rf_TIPI3_auc_value

TIPI3_CF_RF <- confusionMatrix(data = TIPI3_Test$TIPI3, reference=TIPI3_Pred)
TIPI3_CF_RF

TIPI4_RandomForest <- randomForest(TIPI4_Train$TIPI4 ~ ., data=TIPI4_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI4_RandomForest

rfPredTIPI4 <- predict(TIPI4_RandomForest, newdata = TIPI4_Test, type = "prob")[,2]
rfPredTIPI4 <- prediction(rfPredTIPI4, labels = TIPI4_Test$TIPI4)
rf_TIPI4_auc <- performance(rfPredTIPI4, measure = "auc")
rf_TIPI4_auc_value <- rf_TIPI4_auc@y.values
rf_TIPI4_auc_value

TIPI4_CF_RF <- confusionMatrix(data = TIPI4_Test$TIPI4, reference=TIPI4_Pred)
TIPI4_CF_RF

TIPI5_RandomForest <- randomForest(TIPI5_Train$TIPI5 ~ ., data=TIPI5_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI5_RandomForest

rfPredTIPI5 <- predict(TIPI5_RandomForest, newdata = TIPI5_Test, type = "prob")[,2]
rfPredTIPI5 <- prediction(rfPredTIPI5, labels = TIPI5_Test$TIPI5)
rf_TIPI5_auc <- performance(rfPredTIPI5, measure = "auc")
rf_TIPI5_auc_value <- rf_TIPI5_auc@y.values
rf_TIPI5_auc_value

TIPI5_CF_RF <- confusionMatrix(data = TIPI5_Test$TIPI5, reference=TIPI5_Pred)
TIPI5_CF_RF

TIPI6_RandomForest <- randomForest(TIPI6_Train$TIPI6 ~ ., data=TIPI6_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI6_RandomForest

rfPredTIPI6 <- predict(TIPI6_RandomForest, newdata = TIPI6_Test, type = "prob")[,2]
rfPredTIPI6 <- prediction(rfPredTIPI6, labels = TIPI6_Test$TIPI6)
rf_TIPI6_auc <- performance(rfPredTIPI6, measure = "auc")
rf_TIPI6_auc_value <- rf_TIPI6_auc@y.values
rf_TIPI6_auc_value

TIPI6_CF_RF <- confusionMatrix(data = TIPI6_Test$TIPI6, reference=TIPI6_Pred)
TIPI6_CF_RF

TIPI7_RandomForest <- randomForest(TIPI7_Train$TIPI7 ~ ., data=TIPI7_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI7_RandomForest

rfPredTIPI7 <- predict(TIPI7_RandomForest, newdata = TIPI7_Test, type = "prob")[,2]
rfPredTIPI7 <- prediction(rfPredTIPI7, labels = TIPI7_Test$TIPI7)
rf_TIPI7_auc <- performance(rfPredTIPI7, measure = "auc")
rf_TIPI7_auc_value <- rf_TIPI7_auc@y.values
rf_TIPI7_auc_value

TIPI7_CF_RF <- confusionMatrix(data = TIPI7_Test$TIPI7, reference=TIPI7_Pred)
TIPI7_CF_RF

TIPI8_RandomForest <- randomForest(TIPI8_Train$TIPI8 ~ ., data=TIPI8_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI8_RandomForest

rfPredTIPI8 <- predict(TIPI8_RandomForest, newdata = TIPI8_Test, type = "prob")[,2]
rfPredTIPI8 <- prediction(rfPredTIPI8, labels = TIPI8_Test$TIPI8)
rf_TIPI8_auc <- performance(rfPredTIPI8, measure = "auc")
rf_TIPI8_auc_value <- rf_TIPI8_auc@y.values
rf_TIPI8_auc_value

TIPI8_CF_RF <- confusionMatrix(data = TIPI8_Test$TIPI8, reference=TIPI8_Pred)
TIPI8_CF_RF

TIPI9_RandomForest <- randomForest(TIPI9_Train$TIPI9 ~ ., data=TIPI9_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI9_RandomForest

rfPredTIPI9 <- predict(TIPI9_RandomForest, newdata = TIPI9_Test, type = "prob")[,2]
rfPredTIPI9 <- prediction(rfPredTIPI9, labels = TIPI9_Test$TIPI9)
rf_TIPI9_auc <- performance(rfPredTIPI9, measure = "auc")
rf_TIPI9_auc_value <- rf_TIPI9_auc@y.values
rf_TIPI9_auc_value

TIPI9_CF_RF <- confusionMatrix(data = TIPI9_Test$TIPI9, reference=TIPI9_Pred)
TIPI9_CF_RF

TIPI10_RandomForest <- randomForest(TIPI10_Train$TIPI10 ~ ., data=TIPI10_Train, 
                                   ntree=1000, mtry=4, proximity=TRUE, 
                                   importance = TRUE)
TIPI10_RandomForest

rfPredTIPI10 <- predict(TIPI10_RandomForest, newdata = TIPI10_Test, type = "prob")[,2]
rfPredTIPI10 <- prediction(rfPredTIPI10, labels = TIPI10_Test$TIPI10)
rf_TIPI10_auc <- performance(rfPredTIPI10, measure = "auc")
rf_TIPI10_auc_value <- rf_TIPI10_auc@y.values
rf_TIPI10_auc_value

TIPI10_CF_RF <- confusionMatrix(data = TIPI10_Test$TIPI10, reference=TIPI10_Pred)
TIPI10_CF_RF

## --------------------------- -RANDOM FOREST END --------------------------- ##


## -------------------------------- ANN START ------------------------------- ##
library(ROCR)
library(pROC)
#reduce test data to 40 test cases
TIPI1_Test_Modified <- TIPI1_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI1_Train$TIPI1, k = 10)

#training on fold 1
Train <- TIPI1_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

#making predictions
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

#outputs error, reached.threshold and steps
model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI1_Test_Modified$TIPI1, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI1_Test_Modified$TIPI1)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI1_Test_Modified$TIPI1, prob.result)

Train <- TIPI1_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction
Train <- TIPI1_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI1_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

Train <- TIPI1_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI1 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction




##############################
TIPI2_Test_Modified <- TIPI2_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI2_Train$TIPI2, k = 10)

#training on fold 1
Train <- TIPI2_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI2_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI2_Test_Modified$TIPI2, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI2_Test_Modified$TIPI2)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI2_Test_Modified$TIPI2, prob.result)


Train <- TIPI2_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI2_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI2 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)



######################
TIPI3_Test_Modified <- TIPI3_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI3_Train$TIPI3, k = 10)

#training on fold 1
Train <- TIPI3_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI1_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI3_Test_Modified$TIPI3, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI3_Test_Modified$TIPI3)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI3_Test_Modified$TIPI3, prob.result)

Train <- TIPI3_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

Train <- TIPI3_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI3 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)



####################
TIPI4_Test_Modified <- TIPI4_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI4_Train$TIPI4, k = 10)

#training on fold 1
Train <- TIPI4_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI4_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI4_Test_Modified$TIPI4, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI4_Test_Modified$TIPI4)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI4_Test_Modified$TIPI4, prob.result)

Train <- TIPI4_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI4_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI4 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


#######################
TIPI5_Test_Modified <- TIPI5_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI5_Train$TIPI5, k = 10)

#training on fold 1
Train <- TIPI5_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI5_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI5_Test_Modified$TIPI5, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI5_Test_Modified$TIPI5)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI5_Test_Modified$TIPI5, prob.result)


Train <- TIPI5_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI5_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI5 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


############################
TIPI6_Test_Modified <- TIPI6_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI6_Train$TIPI6, k = 10)

#training on fold 1
Train <- TIPI1_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI6_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI6_Test_Modified$TIPI6, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI6_Test_Modified$TIPI6)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI6_Test_Modified$TIPI6, prob.result)

Train <- TIPI1_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI6 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


##########################
TIPI7_Test_Modified <- TIPI7_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI7_Train$TIPI1, k = 10)

#training on fold 1
Train <- TIPI7_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI7_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI7_Test_Modified$TIPI7, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI7_Test_Modified$TIPI7)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI7_Test_Modified$TIPI7, prob.result)

Train <- TIPI7_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI7_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI7 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


#######################
TIPI8_Test_Modified <- TIPI8_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI8_Train$TIPI1, k = 10)

#training on fold 1
Train <- TIPI8_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI8_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI8_Test_Modified$TIPI8, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI8_Test_Modified$TIPI8)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI8_Test_Modified$TIPI8, prob.result)

Train <- TIPI8_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI8_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI8 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


######################
TIPI9_Test_Modified <- TIPI9_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI9_Train$TIPI9, k = 10)

#training on fold 1
Train <- TIPI9_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI19_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI9_Test_Modified$TIPI9, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI9_Test_Modified$TIPI9)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI9_Test_Modified$TIPI9, prob.result)

Train <- TIPI9_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI9_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI9 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)


############################
TIPI10_Test_Modified <- TIPI10_Test[1:40,]
set.seed(150)
#create folds
Folds <- createFolds(TIPI10_Train$TIPI10, k = 10)

#training on fold 1

Train <- TIPI10_Train[Folds$Fold01, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)

ann.results <- compute(model_ANN, TIPI10_Test_Modified)
strengthPrediction <- ann.results$net.result
strengthPrediction

model_ANN$result.matrix

#outputs predicted and actual values
results <- data.frame(actual = TIPI10_Test_Modified$TIPI10, prediction = ann.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

#roc curve 
prob.result <- ann.results$net.result
nn.pred = prediction(prob.result, TIPI10_Test_Modified$TIPI10)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref)
#calculate auc
auc(TIPI10_Test_Modified$TIPI10, prob.result)

Train <- TIPI1_Train[Folds$Fold02, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold03, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold04, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold05, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold06, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold07, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold08, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold09, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)
Train <- TIPI1_Train[Folds$Fold10, ]
model_ANN <- neuralnet(TIPI10 ~ Q1A + Q2A + Q3A + Q4A + Q5A + Q6A + Q7A + Q8A + Q9A +
                         Q10A + Q11A +  Q12A + Q13A + Q14A + Q15A + Q16A + Q17A + Q18A +
                         Q19A + Q20A, data = Train, hidden = 3, linear.output = T)
plot(model_ANN)





