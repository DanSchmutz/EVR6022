# The Gospel of Data Science: Machine Learning and AI 
# 250424
# Dan Schmutz, dschmutz@gpinet.com

# decision trees

# Mandatory package
library(tidyverse)   # includes ggplot2 for plotting and dplyr for data wrangling

# Modeling packages
library(rpart)       # direct engine for decision tree application
library(caret)       # meta engine for decision tree application

# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects

# Load the Titanic dataset
library(titanic)
library(skimr)

library(visdat) # visualize missing data
library(ggpubr)

# what datasets were loaded with titanic?
data()
View(titanic_train)
skim(titanic_train)


vis_miss(titanic_train)

ttrain<-titanic_train
sum(is.na(ttrain))
ttrain[ttrain==""] <- NA
vis_miss(ttrain)
View(ttrain)

ttrain2<- ttrain %>% dplyr::select(-Cabin, -PassengerId, -Ticket)
# Dropping Cabin and PassengerID as not likely to help determine survival

class(ttrain2$Sex)
ttrain2$Gender<-factor(ttrain2$Sex)
ttrain2$Pclass2 <- factor(ttrain2$Pclass, order=TRUE, levels = c(3, 2, 1))
ttrain2$Survived2 <- factor(ttrain2$Survived)
ttrain3<- ttrain2 %>% dplyr::select(-Survived, -Sex, -Pclass)


# Exploratory Data Analysis

ttrain2<- ttrain %>% dplyr::select(-Cabin, -PassengerId, -Ticket)
# Dropping Cabin and PassengerID as not likely to help determine survival

class(ttrain2$Sex)

# making factors
ttrain2$Gender<-factor(ttrain2$Sex)
ttrain2$Pclass2 <- factor(ttrain2$Pclass, order=TRUE, levels = c(3, 2, 1))
ttrain2$Survived2 <- factor(ttrain2$Survived)
ttrain3<- ttrain2 %>% dplyr::select(-Survived, -Sex, -Pclass)


View(ttrain3)


# EDA plots
library(ggpubr)
# Titanic Plot 1
plot_count <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar() +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  ggtitle("Most of the Titanic Passengers are Male.\n 
Most Passengers Who Survived Were Female") +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

plot_percent <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar(position = "fill") +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  ggtitle("75% of all Female Passengers Survived whereas only \n around 20% of the Male Passengers Survived") +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

ggarrange(plot_count, plot_percent)

### Titanic Plot 2
plot_count <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar() +
  facet_wrap(~Pclass2) +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  theme(legend.position = "bottom")

plot_percent <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar(position = "fill") +
  facet_wrap(~Pclass2) +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  theme(legend.position = "bottom") +
  ylab("%")

combined_figure <- ggarrange(plot_count, plot_percent)
annotate_figure(combined_figure,
                top = text_grob("Almost All Female Passengers Who are Class One and Two Survived. The Big Proportion of Men not Surviving \n Mainly Comes From Male Class 3 Passengers",
                                color = "black",
                                face = "bold",
                                size = 14
                )
)


### Titanic Plot 3
plot_count <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar() +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  ggtitle("Most of the Titanic Passengers are Male.\n 
Most Passengers Who Survived Were Female") +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

plot_percent <- ggplot(ttrain3, aes(x = Gender, fill = Survived2)) +
  geom_bar(position = "fill") +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  ggtitle("75% of all Female Passengers Survived whereas only \n around 20% of the Male Passengers Survived") +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

ggarrange(plot_count, plot_percent)

plot_age <- ggplot(ttrain3, aes(x = Age, fill = Survived2)) +
  geom_histogram() +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  theme(legend.position = "bottom")

plot_fare <- ggplot(ttrain3, aes(x = Fare, fill = Survived2)) +
  geom_histogram() +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  theme(legend.position = "bottom")

plot_embarked <- ggplot(ttrain3, aes(x = Embarked, fill = Survived2)) +
  geom_bar() +
  scale_fill_manual(
    name = "Survived",
    values = c("red", "blue"),
    labels = c("No", "Yes"),
    breaks = c("0", "1")
  ) +
  theme(legend.position = "bottom")

ggarrange(plot_age, 
          plot_fare, 
          plot_embarked, common.legend = TRUE, ncol = 3)


# continuing to engineer train data
median(ttrain3$Age, na.rm=T)
ttrain4<- ttrain3 %>% dplyr::select(-Name)
ttrain4$Age[is.na(ttrain4$Age)]<-28
vis_miss(ttrain4)
sum(is.na(ttrain4))
table(ttrain4$Embarked)
ttrain4$Embarked[is.na(ttrain4$Embarked)]<-"S"
vis_miss(ttrain4)


# applying same steps to test
ttest<-titanic_test
ttest[ttest==""] <- NA
ttest2<- ttest %>% dplyr::select(-Cabin, -PassengerId, -Ticket)
ttest2$Gender<-factor(ttest2$Sex)
ttest2$Pclass2 <- factor(ttest2$Pclass, order=TRUE, levels = c(3, 2, 1))
# finding the actual test labels
titanic3 <- read_csv("data/titanic3.csv")
titanicjoinfile<- titanic3 %>% select(name, survived)
ttest2b<-left_join(ttest2,titanicjoinfile,by=c('Name'='name'))
ttest2b$Survived<-ttest2b$survived
ttest2b$Survived2 <- factor(ttest2b$Survived)
ttest3<- ttest2b %>% dplyr::select(-Survived, -survived, -Sex, -Pclass)
ttest4<- ttest3 %>% dplyr::select(-Name)
ttest4$Age[is.na(ttest4$Age)]<-28
ttest4$Embarked[is.na(ttest4$Embarked)]<-"S"
vis_miss(ttest4)
ttest4<- ttest4 %>% drop_na()
vis_miss(ttest4)

# saving copies of the final processed files
write.csv(ttrain4,file='data/ttrain4.csv',row.names=F)
write.csv(ttest4,file='data/ttest4.csv',row.names=F)


# Decision tree modeling

# fitting using class appears appropriate here
rp_ttrain4cl <- rpart(
  formula = Survived2 ~ .,
  data    = ttrain4,
  method  = "class"
)

summary(rp_ttrain4cl)
rpart.plot(rp_ttrain4cl, extra=101)
plotcp(rp_ttrain4cl)
rpart.rules(rp_ttrain4cl, extra=4)

# vip(rp_ttrain4cl, num_features = 40, bar = FALSE)
vip(rp_ttrain4cl)

# Construct partial dependence plots
p1 <- partial(rp_ttrain4cl, pred.var = "Gender") %>% autoplot()
p2 <- partial(rp_ttrain4cl, pred.var = "Fare") %>% autoplot()
p3 <- partial(rp_ttrain4cl, pred.var = "Pclass2") %>% autoplot()
p4 <- partial(rp_ttrain4cl, pred.var = "Age") %>% autoplot() 

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 4)

# two variable partial dependence plot 3D
p5 <- partial(rp_ttrain4cl, pred.var = c("Age", "Fare")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, 
              colorkey = TRUE, screen = list(z = -20, x = -60))
p5

p6 <- partial(rp_ttrain4cl, pred.var = c("Age", "Fare"))
plotPartial(p6)

p7 <- partial(rp_ttrain4cl, pred.var = c("Gender", "Age"))
plotPartial(p7)

p8 <- partial(rp_ttrain4cl, pred.var = c("Age", "Fare", "Gender"))
plotPartial(p8)

# Construct partial dependence plots
p1 <- partial(rp_ttrain4cl, pred.var = "Gender") %>% autoplot()
p2 <- partial(rp_ttrain4cl, pred.var = "Fare") %>% autoplot()
p3 <- partial(rp_ttrain4cl, pred.var = c("Gender", "Fare")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, screen = list(z = -20, x = -60))

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

# what if we purposely overfit the tree?
rp_ttrain4clmany <- rpart(
  formula = Survived2 ~ .,
  data    = ttrain4,
  method  = "class",
  cp = 0.000001
)
rpart.plot(rp_ttrain4clmany)

# evaluating parsimonious and overfit models on train and test
pred_rp_ttrain4cl_train<-predict(rp_ttrain4cl, newdata=ttrain4)
pred_rp_ttrain4cl_test<-predict(rp_ttrain4cl, newdata=ttest4)
pred_rp_ttrain4clmany_train<-predict(rp_ttrain4clmany, newdata=ttrain4)
pred_rp_ttrain4clmany_test<-predict(rp_ttrain4clmany, newdata=ttest4)

hist(pred_rp_ttrain4cl_train) # base r histogram works here

# evaluating results
library(ROCit)
ROCit_obj <- rocit(score=pred_rp_ttrain4cl_train[,2],class=ttrain4$Survived2)
plot(ROCit_obj, YIndex=F, values=T)
ciAUC(ROCit_obj)

ROCit_obj <- rocit(score=pred_rp_ttrain4cl_test[,2],class=ttest4$Survived2)
plot(ROCit_obj, YIndex=F, values=T)
ciAUC(ROCit_obj)

ROCit_obj <- rocit(score=pred_rp_ttrain4clmany_train[,2],class=ttrain4$Survived2)
plot(ROCit_obj, YIndex=F, values=T)
ciAUC(ROCit_obj)

ROCit_obj <- rocit(score=pred_rp_ttrain4clmany_test[,2],class=ttest4$Survived2)
plot(ROCit_obj, YIndex=F, values=T)
ciAUC(ROCit_obj)

# confusion matrices

# confusion matrix for train parsimonious model
predclass<-pred_rp_ttrain4cl_train[,2] %>% 
  data.frame() %>% 
  mutate(predclass = ifelse(. > 0.5,1,0)) %>% 
  select(predclass)
  
confusionMatrix(factor(predclass$predclass),ttrain4$Survived2)

# confusion matrix for test parsimonious model
predclass<-pred_rp_ttrain4cl_test[,2] %>% 
  data.frame() %>% 
  mutate(predclass = ifelse(. > 0.5,1,0)) %>% 
  select(predclass)

confusionMatrix(factor(predclass$predclass),ttest4$Survived2)

# confusion matrix for train overfit model
predclass<-pred_rp_ttrain4clmany_train[,2] %>% 
  data.frame() %>% 
  mutate(predclass = ifelse(. > 0.5,1,0)) %>% 
  select(predclass)

confusionMatrix(factor(predclass$predclass),ttrain4$Survived2)

# confusion matrix for test overfit model
predclass<-pred_rp_ttrain4clmany_test[,2] %>% 
  data.frame() %>% 
  mutate(predclass = ifelse(. > 0.5,1,0)) %>% 
  select(predclass)

confusionMatrix(factor(predclass$predclass),ttest4$Survived2)



### random forest

library(randomForest)
library(caret)

# extra processing to avoid error
ttrain4rf<-ttrain4  %>% 
  mutate(Survived2 = factor(Survived2, 
                            labels = make.names(levels(Survived2))))

ttest4rf<-ttest4  %>% 
  mutate(Survived2 = factor(Survived2, 
                            labels = make.names(levels(Survived2))))


# code below used to identify optimal mtry hyperparameter for tuning using caret
seed<-42
set.seed(seed)
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid", classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(1:7))
metric<-"Accuracy"
rf_gridsearch <- train(Survived2~., data= ttrain4rf, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
print(rf_gridsearch)

set.seed(9)
rfm_ttrain4_500<-randomForest(Survived2~.,ttrain4rf,mtry=3,ntree=500) # perform random forest using 80% training dataset and optimized hyperparameters

print(rfm_ttrain4_500) # view OOB estimate of error rate
plot(rfm_ttrain4_500)

set.seed(9)
rfm_ttrain4_10000<-randomForest(Survived2~.,ttrain4rf,mtry=3,ntree=10000) # perform random forest using 80% training dataset and optimized hyperparameters

print(rfm_ttrain4_10000) # view OOB estimate of error rate, no improvement over 500 trees
plot(rfm_ttrain4_10000)


pred_rfm_500_test<-predict(rfm_ttrain4_500,newdata=ttest4rf) # make predictions on 20% test dataset using training dataset-developed model
confusionMatrix(pred_rfm_500_test,ttest4rf$Survived2) # evaluating performance on out-of-sample test dataset 

pred_rfm_10000_test<-predict(rfm_ttrain4_10000,newdata=ttest4rf) 
confusionMatrix(pred_rfm_10000_test,ttest4rf$Survived2) 

rfm_ttrain4_500i<-randomForest(Survived2~.,ttrain4rf,mtry=3,ntree=500, importance=T)

varImpPlot(rfm_ttrain4_500i) # visualize variable importance

# Construct partial dependence plots
p11 <- partial(rfm_ttrain4_500i, pred.var = "Gender") %>% autoplot()
p12 <- partial(rfm_ttrain4_500i, pred.var = "Pclass2") %>% autoplot()
p13 <- partial(rfm_ttrain4_500i, pred.var = "Fare") %>% autoplot()
p14 <- partial(rfm_ttrain4_500i, pred.var = "Age") %>% autoplot() 

# Display plots side by side
gridExtra::grid.arrange(p11, p12, p13, p14, ncol = 4)


# comparing multiple models using caret

library(caret)
library(mda)
library(import)
library(xgboost)



control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(42)
model_rf <- train(Survived2~., data=ttrain4rf, method="rf", trControl=control, verbose=FALSE)
set.seed(42)
model_blr <- train(Survived2~., data=ttrain4rf, method="LogitBoost", trControl=control, verbose=FALSE)
set.seed(42)
model_xgb <- train(Survived2~., data=ttrain4rf, method="xgbTree", trControl=control, verbose=FALSE)

results_mc <- resamples(list(RF=model_rf, LBOOST=model_blr, XGBoost=model_xgb))
summary(results_mc)
bwplot(results_mc)
dotplot(results_mc)

# compare predictions on test
pred_model_rf_test_tknn<-predict(model_rf,newdata=ttest4rf)
pred_model_blr_test_tknn<-predict(model_blr,newdata=ttest4rf)
pred_model_xgb_test_tknn<-predict(model_xgb,newdata=ttest4rf)
confusionMatrix(pred_model_rf_test_tknn,ttest4rf$Survived2)
confusionMatrix(pred_model_blr_test_tknn,ttest4rf$Survived2)
confusionMatrix(pred_model_xgb_test_tknn,ttest4rf$Survived2)

# Ames Housing data
# Ames Housing data

libraries
library(AmesHousing)
library(rsample)
library(caret)

# access data
ames <- AmesHousing::make_ames()

# initial dimension
dim(ames)
## [1] 2930   81

colnames(ames)

# response variable
head(ames$Sale_Price)
## [1] 215000 105000 172000 244000 189900 195500

# data partitions

# Using base R
set.seed(42)  # for reproducibility
index_1 <- sample(1:nrow(ames), round(nrow(ames) * 0.8))
train_1 <- ames[index_1, ]
test_1  <- ames[-index_1, ]

# Using caret package
set.seed(42)  # for reproducibility
index_2 <- createDataPartition(ames$Sale_Price, p = 0.8, list = FALSE)
train_2 <- ames[index_2, ]
test_2  <- ames[-index_2, ]

library(tidyverse)
library(ggplot2)
library(reshape2)
library(rsample)
library(reshape)

# have to use list in the next one because of different lengths of the variables, dataframe won't work
m1 <- list(train_1=train_1$Sale_Price,test_1=test_1$Sale_Price,train_2=train_2$Sale_Price, test_2=test_2$Sale_Price)
m1m<- melt(m1)
ggplot(m1m,aes(x=value,color=L1)) + geom_density(alpha=0.5)
ggplot(m1m,aes(x=value, fill=L1)) + geom_boxplot()

# stratified sampling
library(rsample)
index_3 <- initial_split(ames,0.8, strata = "Sale_Price", breaks=4)
train_3 <- training(index_3)
test_3  <- testing(index_3)
m2 <- list(train_3=train_3$Sale_Price,test_3=test_3$Sale_Price)
m2m<- melt(m2)
ggplot(m2m,aes(x=value,color=L1)) + geom_density(alpha=0.5)
ggplot(m2m,aes(x=value, fill=L1)) + geom_boxplot()


# saving copies of the final processed files to use in other projects
write.csv(train_3,file='data/train_3.csv',row.names=F)
write.csv(test_3,file='data/test_3.csv',row.names=F)


# target engineering and arranging multiple plots
library(gridExtra)
ggplot(train_3, aes(x=Sale_Price))+geom_histogram(color='white')
ggplot(train_3, aes(x=log(Sale_Price)))+geom_histogram(color='white')

# histogram plot comparison of regular and logged sales
plot_1<-ggplot(train_3, aes(x=Sale_Price))+geom_histogram(color='white')+labs(title='Sale Price')
plot_2<-ggplot(train_3, aes(x=log(Sale_Price)))+geom_histogram(color='white')+labs(title='Log Sale Price')
grid.arrange(plot_1,plot_2, ncol=1)

# normal probability quantile plots       
plot_1b <- ggplot(train_3, aes(sample = Sale_Price))+ stat_qq() + stat_qq_line()+labs(title='Sale Price')
plot_1b
plot_2b <- ggplot(train_3, aes(sample = log(Sale_Price)))+ stat_qq() + stat_qq_line()+labs(title='Log Sale Price')
plot_2b
grid.arrange(plot_1b,plot_2b, ncol=2)

shapiro.test(train_3$Sale_Price)
shapiro.test(log(train_3$Sale_Price))


# Box Cox transform
library(recipes) # allows pre-processing of variables prior to modeling
simple_trans_rec <- recipe(Sale_Price ~ ., data = train_3) %>%
  step_BoxCox(Sale_Price) %>%
  prep(training = train_3)

simple_trans_result <- bake(simple_trans_rec, train_3)

library(EnvStats) # Box Cox in EnvStats
box_1<-boxcox(train_3$Sale_Price, optimize=T)
print(box_1)

# Box Cox using the package forecast 
library(forecast)
box_2<-BoxCox.lambda(train_3$Sale_Price)
box_2
train_3t<-BoxCox(train_3$Sale_Price,lambda= "auto")
train_3t_df<-data.frame(train_3t)
colnames(train_3t_df)[1]<-"Box_Cox_Sale_Price"

plot_1<-ggplot(train_3, aes(x=Sale_Price))+
  geom_histogram(color='white')+labs(title='Sale Price')
plot_2<-ggplot(train_3, aes(x=log(Sale_Price)))+
  geom_histogram(color='white')+labs(title='Log Sale Price')
plot_3<-ggplot(train_3t_df, aes(x=Box_Cox_Sale_Price))+
  geom_histogram(color='white')+labs(title='Box Cox Sale Price')
grid.arrange(plot_1,plot_2,plot_3, ncol=1)
# InvBoxCox(x, lambda, biasadj = FALSE, fvar = NULL) will get  variable back

# are any values missing?
is.na(ames) %>% table()
sum(is.na(ames))

# modeling with Ames
# working on linear models with ames



# Model interpretability packages
library(vip)      # variable importance
library(visreg) # visualizing partial residual plots

# basic linear regression
lm1 <- lm(Sale_Price ~ Gr_Liv_Area, data = train_3)
summary(lm1)

ggplot(train_3, aes(x= Gr_Liv_Area, y=Sale_Price))+
  geom_point()+stat_smooth(method="lm", se=F)

# saving training set prediction and residuals for plotting
t3a<-train_3
t3a$predicted<-predict(lm1)
t3a$resid<-residuals(lm1)
ggplot(t3a, aes(x= Gr_Liv_Area, y=Sale_Price))+
  geom_point()+
  stat_smooth(method="lm", se=F)+
  geom_point(aes(y = predicted), shape = 1)+
  geom_segment(aes(xend = Gr_Liv_Area, yend = predicted), alpha=0.75, color='magenta')


sigma(lm1)    # RMSE same as residual standard error in output (with rounding)
## [1] 53992.84
sigma(lm1)^2  # MSE
## [1] 2915226792

confint(lm1, level = 0.95)
##             2.5 %     97.5 %
##(Intercept)  794.7637 14583.3121
## Gr_Liv_Area 111.0540   119.7955

# mutliple linear regression, two variables
lm2 <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built, data = train_3)
summary(lm2)
coef(lm2)
sigma(lm2)

visreg2d(lm2, "Gr_Liv_Area", "Year_Built")

# mutliple linear regression, allowing interaction
lm2b <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built + Gr_Liv_Area:Year_Built, data = train_3)
summary(lm2b)
coef(lm2b)
sigma(lm2b)
visreg2d(lm2b, "Gr_Liv_Area", "Year_Built")

lm3 <- lm(Sale_Price ~ ., data = train_3)
sigma(lm3)
summary(lm3)
visreg(lm3) # let's visualize a few of the variable partial residuals


# using caret's cv to compare models using 10-fold cv
# Train model using 10-fold cross-validation
# model 1 CV
set.seed(42)  # for reproducibility
cv_model1 <- train(
  form = Sale_Price ~ Gr_Liv_Area, 
  data = train_3, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)
# model 2 CV
set.seed(42)
cv_model2 <- train(
  Sale_Price ~ Gr_Liv_Area + Year_Built, 
  data =train_3, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# model 3 CV
set.seed(42)
cv_model3 <- train(
  Sale_Price ~ ., 
  data = train_3, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# Extract out of sample performance measures
sum123<-summary(resamples(list(
  model1 = cv_model1, 
  model2 = cv_model2, 
  model3 = cv_model3
)))
sum123

#better

results123<-resamples(list(
  model1 = cv_model1, 
  model2 = cv_model2, 
  model3 = cv_model3
))

summary(results123)
bwplot(results123)
dotplot(results123)


# adding residuals to examine assumptions

cv_model1$finalModel
cv_model3$finalModel

dflm1 <- broom::augment(cv_model1$finalModel, data = train_3)
str(dflm1)
glimpse(dflm1)

dflm3 <- broom::augment(cv_model3$finalModel, data = train_3)

p1lm1 <- ggplot(dflm1, aes(.fitted, .std.resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Sale_Price ~ Gr_Liv_Area")
p1lm1

p1lm3 <- ggplot(dflm3, aes(.fitted, .std.resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Sale_Price ~ .")
p1lm3

grid.arrange(p1lm1,p1lm3,ncol=2)

# are the residuals correlated by row number?
dflm1 <- mutate(dflm1, id = row_number())
dflm3 <- mutate(dflm3, id = row_number())

p1 <- ggplot(dflm1, aes(id, .std.resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Correlated residuals.")+
  stat_smooth(method='loess',span=0.1)

p2 <- ggplot(dflm3, aes(id, .std.resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Uncorrelated residuals.")+
  stat_smooth(method='loess',span=0.1)

gridExtra::grid.arrange(p1, p2, nrow = 1)

library(corrplot)
# look at correlations among the 35 numeric variables
train_3 %>% dplyr::select_if(is.numeric) %>% 
  cor() %>% corrplot()

# how do we do on prediction with the 3 models, evaluated on the test data?
pred_lm1<-predict(lm1,newdata=test_3)
pred_lm2<-predict(lm2,newdata=test_3)
pred_lm3<-predict(lm3,newdata=test_3) # problem with new level in roof material showing up in test set

test_3aug<-test_3
test_3aug$pred_lm1<-pred_lm1
test_3aug$pred_lm2<-pred_lm2

p70<-ggplot(test_3aug,aes(x=pred_lm1,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+
  geom_abline(slope=1, intercept=0, col='red')
p71<-ggplot(test_3aug,aes(x=pred_lm2,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+
  geom_abline(slope=1, intercept=0, col='red')
gridExtra::grid.arrange(p70, p71, nrow = 1)

cor(test_3aug$pred_lm1,test_3aug$Sale_Price)^2
cor(test_3aug$pred_lm2,test_3aug$Sale_Price)^2

res_lm1<-test_3aug$Sale_Price-test_3aug$pred_lm1
res_lm2<-test_3aug$Sale_Price-test_3aug$pred_lm2

(mean((res_lm1)^2))^0.5
(mean((res_lm2)^2))^0.5



#trying again with smaller dataset
train_3num<-train_3 %>% dplyr::select_if(is.numeric)


set.seed(42)  # for reproducibility
cv_modelglm4 <- train(
  form = Sale_Price ~ Lot_Frontage+Lot_Area+Year_Built+Year_Remod_Add+Mas_Vnr_Area+Bsmt_Unf_SF+Total_Bsmt_SF+First_Flr_SF+Second_Flr_SF+Bsmt_Full_Bath+Bedroom_AbvGr+Kitchen_AbvGr+TotRms_AbvGrd+Fireplaces+Garage_Cars+Wood_Deck_SF+Pool_Area+Latitude, 
  data = train_3num_sm, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

pred_lmglm4<-predict(cv_modelglm4$finalModel,newdata=test_3)

test_3aug$pred_lmglm4<-pred_lmglm4
ggplot(test_3aug,aes(x=pred_lmglm4,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+geom_abline(slope=1, intercept=0, col='red')
cor(test_3aug$pred_lmglm4,test_3aug$Sale_Price)^2
res_lmglm4<-test_3aug$Sale_Price-test_3aug$pred_lmglm4
(mean((res_lmglm4)^2))^0.5

# applying other ML methods to ames


# glmnet regression
library(tidyverse)

# Helper packages
library(recipes)  # for feature engineering

# Modeling packages
library(glmnet)   # for implementing regularized regression
library(caret)    # for automating the tuning process

# Model interpretability packages
library(vip)      # for variable importance


# Create training feature matrices
# unfortuntately have to use a model.matrix instead of the usual dataframe
# we use model.matrix(...)[, -1] to discard the intercept
X <- model.matrix(Sale_Price ~ ., train_3)[, -1]
Xnotused<-model.matrix(Sale_Price ~.,train_3)

# transform y with log transformation
Y <- log(train_3$Sale_Price)

# Apply ridge regression to ames data
ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

plot(ridge, xvar = "lambda")


str(ridge) # what does the object look like

# lambdas applied to penalty parameter
ridge$lambda %>% head()
## [1] 285.8055 260.4153 237.2807 216.2014 196.9946 179.4942

# small lambda results in large coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 100]
##                   Latitude Overall_QualVery_Excellent 
##                  0.4048216                  0.1423770

# large lambda results in small coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 1]  
##                                      Latitude 
## 0.0000000000000000000000000000000000063823847 
##                    Overall_QualVery_Excellent 
## 0.0000000000000000000000000000000000009838114

ridge$lambda[2]
ridge$lambda[99]

# Apply CV ridge regression to ames data
ridge <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 0
)
plot(ridge, main = "Ridge penalty\n\n")

# Ridge model
min(ridge$cvm)       # minimum MSE
## [1] 0.01968996
ridge$lambda.min     # lambda for this min MSE
## [1] 0.1417314

ridge$cvm[ridge$lambda == ridge$lambda.1se]  # 1-SE rule
## [1] 0.01975572
ridge$lambda.1se  # lambda for this MSE
## [1] 0.6279583

ridge$lambda.min
#[1] 0.1417314
log(ridge$lambda.min)
#[1] -1.953821
ridge$lambda.1se
#[1] 0.6279583
log(ridge$lambda.1se)
#[1] -0.4652815


# evaluation on train_3 ridge
predridge <- predict(ridge, X)
# compute RMSE of transformed predicted
RMSE(exp(predridge), exp(Y))

# predicting test_3 using ridge
Xtest <- model.matrix(Sale_Price ~ ., test_3)[, -1]
Ytest <- log(test_3$Sale_Price)
predridgetest <- predict(ridge, Xtest)
RMSE(exp(predridgetest), exp(Ytest))

test_3aug$predridgetest<-exp(predridgetest)
ggplot(test_3aug,aes(x=predridgetest,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+geom_abline(slope=1, intercept=0, col='red')
cor(test_3aug$predridgetest,test_3aug$Sale_Price)^2
res_predridgetest<-test_3aug$Sale_Price-test_3aug$predridgetest
(mean((res_predridgetest)^2))^0.5


#lasso on ames data

lasso <- glmnet(
  x = X,
  y = Y,
  alpha = 1
)

plot(lasso, xvar = "lambda")


# Apply CV lasso regression to Ames data
lasso <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 1
)
plot(lasso, main = "Lasso penalty\n\n")

# Lasso model
min(lasso$cvm)       # minimum MSE
# [1] 0.02046739
lasso$lambda.min     # lambda for this min MSE
# [1] 0.002782246
lasso$nzero[lasso$lambda == lasso$lambda.min] # No. of coef | Min MSE
#s50  137
lasso$cvm[lasso$lambda == lasso$lambda.1se]  # 1-SE rule
#[1] 0.02294067
lasso$lambda.1se  # lambda for this MSE
# [1] 0.01232708
lasso$nzero[lasso$lambda == lasso$lambda.1se] # No. of coef | 1-SE MSE
# s34 57


lasso$lambda.min
#[1] 0.002535079
log(lasso$lambda.min)
#[1] -5.977531
lasso$lambda.1se
#0.01352895
log(lasso$lambda.1se)
#-4.302923


# evaluation on train_3 lasso
predlasso <- predict(lasso, X)
# compute RMSE of transformed predicted
RMSE(exp(predlasso), exp(Y))

# predicting test_3 using lasso
Xtest <- model.matrix(Sale_Price ~ ., test_3)[, -1]
Ytest <- log(test_3$Sale_Price)
predlassotest <- predict(lasso, Xtest)
RMSE(exp(predlassotest), exp(Ytest))

test_3aug$predlassotest<-exp(predlassotest)
ggplot(test_3aug,aes(x=predlassotest,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+geom_abline(slope=1, intercept=0, col='red')
cor(test_3aug$predlassotest,test_3aug$Sale_Price)^2
res_predlassotest<-test_3aug$Sale_Price-test_3aug$predlassotest
(mean((res_predlassotest)^2))^0.5


# elastic net model
# for reproducibility
set.seed(42)

# grid search across 
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# model with lowest RMSE
cv_glmnet$bestTune
##   alpha     lambda
## 8   0.1 0.04728445

# results for model with lowest RMSE
cv_glmnet$results %>%
  filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)
##   alpha     lambda      RMSE  Rsquared        MAE    RMSESD RsquaredSD
## 1   0.1 0.04728445 0.1382785 0.8852799 0.08427956 0.0273119 0.04179193
## MAESD
## 1 0.005330001

# plot cross-validated RMSE
ggplot(cv_glmnet)

# evaluation on train_3 elastic net
predelastic <- predict(cv_glmnet, X)
# compute RMSE of transformed predicted
RMSE(exp(predelastic), exp(Y))

# predicting test_3 using elastic net
Xtest <- model.matrix(Sale_Price ~ ., test_3)[, -1]
Ytest <- log(test_3$Sale_Price)
predelastictest <- predict(cv_glmnet, Xtest)
RMSE(exp(predelastictest), exp(Ytest))

test_3aug$predelastictest<-exp(predelastictest)
ggplot(test_3aug,aes(x=predelastictest,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+geom_abline(slope=1, intercept=0, col='red')
cor(test_3aug$predelastictest,test_3aug$Sale_Price)^2
res_predelastictest<-test_3aug$Sale_Price-test_3aug$predelastictest
(mean((res_predelastictest)^2))^0.5

# variable importance plot
vip(cv_glmnet, num_features = 20, geom = "point")


# comparing performance to a random forest regression
# random forest on ames data
m_rf_ames_v1<-randomForest(Sale_Price~., data=train_3)
print(m_rf_ames_v1)
plot(m_rf_ames_v1)

# warning code commented out below took more than 30 min so terminated process
# code below used to identify optimal mtry hyperparameter for tuning using caret

# seed<-42
# set.seed(seed)
# control <- trainControl(method="cv", number=10, search="grid")
# tunegrid <- expand.grid(.mtry=c(1:80))
# metric<-"RMSE"
# rf_gridsearch <- train(Sale_Price~., data= train_3, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
# print(rf_gridsearch)
# plot(rf_gridsearch)
# print(rf_gridsearch)

library(randomForest)
m_rf_ames_v1<-randomForest(Sale_Price~., data=train_3, importance=T)
print(m_rf_ames_v1)
plot(m_rf_ames_v1)

pred_rf_ames_v1<-predict(m_rf_ames_v1,newdata=test_3)

RMSE(m_rf_ames_v1$predicted, train_3$Sale_Price)
RMSE(pred_rf_ames_v1, test_3$Sale_Price)

test_3aug$predrf1<-pred_rf_ames_v1
ggplot(test_3aug,aes(x=predrf1,y=Sale_Price))+
  geom_point()+stat_smooth(method=lm)+geom_abline(slope=1, intercept=0, col='red')
cor(test_3aug$predrf1,test_3aug$Sale_Price)^2
res_predrf1<-test_3aug$Sale_Price-test_3aug$predrf1
(mean((res_predrf1)^2))^0.5

# exploring the variable importance
varImpPlot(m_rf_ames_v1)

# unsupervised learning

# iris is in the built-in datasets package
data()

irisb<-iris

# prepare to plot
iris2 <- irisb[,-5]
species_labels <- iris[,5]
library(colorspace) # get nice colors
species_col <- rev(rainbow_hcl(3))[as.numeric(species_labels)]


# Plot a SPLOM:
pairs(iris2, col = species_col,
      lower.panel = NULL,
      cex.labels=2, pch=19, cex = 1.2)

# Add a legend
par(xpd = TRUE)
legend(x = 0.05, y = 0.4, cex = 2,
       legend = as.character(levels(species_labels)),
       fill = unique(species_col))
par(xpd = NA)

# alternative using prompt to ChatGPT
library(plotly)
library(GGally)

# Basic scatterplot matrix with GGally
p <- ggpairs(iris, columns = 1:4, aes(color = Species))

# Make it interactive with plotly
ggplotly(p)


# hierarchical clustering based on https://cran.r-project.org/web/packages/dendextend/vignettes/Cluster_Analysis.html
d_iris <- dist(irisb) # method="man" # is a bit better
hc_iris <- hclust(d_iris, method = "average")
iris_species <- rev(levels(iris[,5]))

library(dendextend)
dend <- as.dendrogram(hc_iris)
# order it the closest we can to the order of the observations:
dend <- rotate(dend, 1:150)

# Color the branches based on the clusters:
dend <- color_branches(dend, k=3) #, groupLabels=iris_species)

# Manually match the labels, as much as possible, to the real classification of the flowers:
labels_colors(dend) <-
  rainbow_hcl(3)[sort_levels_values(
    as.numeric(iris[,5])[order.dendrogram(dend)]
  )]

# We shall add the flower type to the labels:
labels(dend) <- paste(as.character(iris[,5])[order.dendrogram(dend)],
                      "(",labels(dend),")", 
                      sep = "")
# We hang the dendrogram a bit:
dend <- hang.dendrogram(dend,hang_height=0.1)
# reduce the size of the labels:
# dend <- assign_values_to_leaves_nodePar(dend, 0.5, "lab.cex")
dend <- set(dend, "labels_cex", 0.5)
# And plot:
par(mar = c(3,3,3,7))
plot(dend, 
     main = "Clustered Iris data set
     (the labels give the true flower species)", 
     horiz =  TRUE,  nodePar = list(cex = .007))
legend("topleft", legend = iris_species, fill = rainbow_hcl(3))

# k-means clustering from factoextra package
library(factoextra)
# Remove species column (5) and scale the data
iris.scaled <- scale(irisb[, -5])

# Optimal number of clusters in the data
# ++++++++++++++++++++++++++++++++++++++
# Examples are provided only for kmeans, but
# you can also use cluster::pam (for pam) or
#  hcut (for hierarchical clustering)

### Elbow method (look at the knee)
# Elbow method for kmeans
fviz_nbclust(iris.scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

# Average silhouette for kmeans
fviz_nbclust(iris.scaled, kmeans, method = "silhouette")

### Gap statistic
library(cluster)
set.seed(123)
# Compute gap statistic for kmeans
# we used B = 10 for demo. Recommended value is ~500
gap_stat <- clusGap(iris.scaled, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 10)
print(gap_stat, method = "firstmax")
fviz_gap_stat(gap_stat)

# selecting k=3
kmeans3<-kmeans(iris.scaled, centers=3, iter.max = 10, nstart = 10)
kmeans3
fviz_cluster(kmeans3,data=iris.scaled)

# Principal Components Analysis
# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/

library(factoextra)
row.names(irisb)
irisb$Species
res.pca <- prcomp(irisb[,-5], scale = TRUE)
fviz_eig(res.pca)

fviz_pca_ind(res.pca,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# the_end_in_snake_case






