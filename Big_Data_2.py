import finspark
finspark.init()
import pyspark
from pyspark import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import isnull, when, count, col
from pyspark.sql.functions import mean, avg
from pyspark.sql.functions import col, trim, lower
from pyspark.sql import Row, SQLContext
from pyspark.sql.functions import when, lit
from pyspark.sql.functions import *
import pyspark.sql.functions as f
#
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, OneHotEncoderModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes, RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import Bucketizer
#
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
#
import pandas as pd
import numpy as np
import os
import sys
#
# Coursework 2: Big Data Analysis March 2022
# Income prediction in Pyspark using Ml and MLlib
# Predict whether income is <=50k or >50k : the UCI Adult Census dataset 
# 
# Create a SparkSession for appName "income" 
sc = SparkSession.builder.master("local[1]")\
.appName("income")\
.getOrCreate()
# Load the dataset into a Spark DataFrame
df_adult = sc.read.csv("hdfs:///user/rtyle001/CW2/data/adult_data.csv",\
                 header=True,\
                       inferSchema=True,\
                       nanValue="?")
#
# print the dataframe schema
df_adult.printSchema()     
df_adult.show(5, truncate = False)
#
# Drop fnlwgt and "x" columns
df=df_adult.drop("fnlwgt", "x")
df.printSchema()
#
# Cast continuous (numeric) feature columns to FloatTYpe columns
columns_to_cast = ["age","capital-gain",\
                 "educational-num","capital-loss",\
                 "hours-per-week"]
df_cast= (df.select(*(c for c in df.columns if c not in columns_to_cast),\
                     *(col(c).cast("float").alias(c) for c in columns_to_cast)))

df_cast.printSchema()
#
df =df_cast
#
# Exploratory Data Analysis
#
# Summary statistics
for col in df.columns:
    df.describe([col]).show()


numeric_features = [t[0] for t in df.dtypes if t[1] == "float"]
numeric_data = df.select(numeric_features).toPandas()
# Correlation of numeric features
numeric_data.corr()
#
df.groupby('marital-status').agg({'capital-gain': 'mean'}).show()
df.groupby('occupation').agg({'capital-gain': 'mean'}).show()
df.groupby('income').agg({'educational-num': 'mean'}).show()
df.groupby('income').agg({'hours-per-week': 'mean'}).show()
df.groupby('race').count().show(truncate=False)
df.groupby('workclass').count().show(truncate=False)
df.groupby('native-country').count().show(truncate=False)
df.groupby('occupation','gender','income').count().show(truncate=False)
#
tot = df.count()
df.groupBy('occupation', 'income').count() \
.withColumnRenamed('count', 'cnt_per_group') \
.withColumn('perc_of_count_total', (f.col('cnt_per_group') / tot) * 100 ) \
.show()

df.groupBy('gender','income').count() \
.withColumn('%', f.round((f.col('count')/tot)*100,2)) \
.orderBy('count', ascending=False) \
.show()

df.groupBy('marital-status','income').count() \
.withColumn('%', f.round((f.col('count')/tot)*100,2)) \
.orderBy('count', ascending=False) \
.show()

df.groupBy('race','income').count() \
.withColumn('%', f.round((f.col('count')/tot)*100,2)) \
.orderBy('count', ascending=False) \
.show()

df.groupBy('education','income').count() \
.withColumn('%', f.round((f.col('count')/tot)*100,2)) \
.orderBy('count', ascending=False) \
.show()

df.groupBy('native-country').count() \
.withColumn('%', f.round((f.col('count')/tot)*100,2)) \
.orderBy('count', ascending=False) \
.show()
#
# print count of row and columns
print("There are {} columns and {} rows in the adult dataset."\
      .format(len(df.columns), df.count()))
#
# Missing data
# Count the value of '?' in every column
for col in df.columns:
    print(col, "\t", "with '?' values: ",\
          df.filter(df[col]=="?").count())
    

#
# Replace entries with "?" with "None"
df2=df.replace("?",None)
# Do a validation Count the value of '?' in every column
for col in df2.columns:
    print(col, "\t", "with '?' values: ",\
          df2.filter(df2[col]=="?").count())

    
# Count the number of null values in every column
for col in df2.columns:
    print(col, "\t", "with null values: ",\
          df2.filter(df2[col].isNull()).count())
    

# Drop null values
df_no_null =df2.na.drop()
#
# Validate there are no nulls
for col in df_no_null.columns:
    print(col, "\t", "with null values: ",\
          df_no_null.filter(df_no_null[col].isNull()).count())
    

#
# Bucketizer
# Group by age
df.groupby("age").count().orderBy("age",ascending=True).show()
df.agg({"age":"min"}).show()
df.agg({"age":"max"}).show()
splits = [17, 30, 50, 70, 90]
bucketizer  = Bucketizer( splits=splits,inputCol="age", outputCol="age_group")
df_age = bucketizer.transform(df)
df_age.select("age_group").show()
#
# Create a new feature for "age" column - add a square to age
from pyspark.sql.functions import col, trim, lower
df_calc = df_no_null.withColumn("age-squared",col("age")**2)
df_calc.printSchema()
#
# Count and sort by country
df_calc.groupby("native-country")\
.agg({"native-country":"count"})\
.sort(asc("count(native-country)")).show()
# Holand-Netherlands only has 1 record
# Filter it out of the dataframe
df3=df_calc.filter(df_calc["native-country"]!="Holand-Netherlands")
#
# compute count of instances per label: class 0: <= 50k income; class 1: > 50k income
class_counts = df3.groupBy("income")\
                              .count()\
                              .collect()
#
# add the counts to a dictionary for display
class_counts_dict = {row["income"]: row['count'] for row in class_counts}
class_counts_dict
#
# print count of row and columns 
print("There are {} columns and {} rows in the adult dataset."\
      .format(len(df3.columns), df3.count()))

# there are 15 columns and 45221 rows
#
# Split the data into training and testing datasets
# Split 80/20
train, test = df3.randomSplit([.8,.2], seed=13)
#
# Category Indexing with StringIndexer
# Categorical columns
cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]
# Numerical columns
num_cols=["age","capital-gain",\
                 "educational-num","capital-loss",\
                 "hours-per-week","age-squared"]
# String Indexer
stringIndexer = StringIndexer(inputCols=cat_cols,\
                              outputCols=[x + "Index" for x in cat_cols])\
.setHandleInvalid("skip")
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(),\
                        outputCols=[x + "OHE" for x in cat_cols])
# The label column ("income") is also a string value - it has two possible values, "<=50K" and ">50K". 
# Convert it to a numeric value using StringIndexer.
labelToIndex = StringIndexer(inputCol="income",\
                             outputCol="label")\
.setHandleInvalid("skip")
stringIndexerModel = stringIndexer.fit(df3)
stringIndexerModel.transform(df3).show(5)
#
# Vector Assembler
assemblerInputs = [c + "OHE" for c in cat_cols] + num_cols
assembler= VectorAssembler(inputCols=assemblerInputs,\
                               outputCol="features")
#
# Logistic Regression Model
# Create the lr model
lr = LogisticRegression(featuresCol="features",\
                        labelCol="label",\
                       maxIter=10)
#
# Create a pipeline for lr model
lr_pipeline= Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 lr])
#
# fit the lr model to training data
lr_pipeline = lr_pipeline.fit(train)
#
# transform the lr model to test data
lr_predict = lr_pipeline.transform(test)
#
# PrintSchema of lr and mlr predictions
lr_predict.printSchema()
#
# Display the predictions for lr and mlr models
lr_select = lr_predict.select("label", "prediction", "probability").show(5)
# Confusion Matrix
lr_cm_predict = lr_predict.crosstab("prediction", "label").show()
#
# Evaluate the lr model
# Calulate the area under the curve 
lr_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(lr_eval.evaluate(lr_predict))
print(lr_eval.getMetricName())
#
# Calculate the f1 score
lr_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(lr_eval2.evaluate(lr_predict))
print(lr_eval2.getMetricName())
#
# Calculate accuracy
lr_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(lr_eval3.evaluate(lr_predict))
print(lr_eval3.getMetricName())
#
# Save the lr pipeline model
lr_pipeline_path = "hdfs:///user/rtyle001/CW2/data/lr_pipeline"
lr_pipeline.write().overwrite().save(lr_pipeline_path)
#
# Load the lr pipeline model
lr_pipeline_model = lr_pipeline.load(lr_pipeline_path)
#
# Hyper-Parameter tuning: Logistic Regression
# Create the model
lr_cv=LogisticRegression(featuresCol="features",\
                            labelCol="label")
#
# Create ParamGrid for cross validataion
lr_cv_paramGrid=ParamGridBuilder()\
.addGrid(lr.regParam,[0.1, 0.5, 2.0])\
.addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
.addGrid(lr.maxIter,[1,5,10])\
.build()
#
lr_cv=CrossValidator(estimator=lr_cv,\
                     estimatorParamMaps=lr_cv_paramGrid,\
                     evaluator=BinaryClassificationEvaluator(),\
                     numFolds=3)
#
# Create a pipeline for lr_cv model
lr_cv_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 lr_cv])
#
# fit the lr model to training data
lr_cv_pipeline = lr_cv_pipeline.fit(train)
#
# transform the lr model to test data
lr_cv_predict = lr_cv_pipeline.transform(test)
#
# Predictions
lr_cv_predict.printSchema()
lr_cv_select = lr_cv_predict.select("label","prediction","probability").show(5)
#
# Evaluate the lr_cv model
# Calculate area under curve
lr_cv_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(lr_cv_eval.evaluate(lr_cv_predict))
print(lr_cv_eval.getMetricName())
#
# Calculate the f1 score
lr_cv_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
lr_cv_eval2.evaluate(lr_cv_predict)
lr_cv_eval2.getMetricName()
#
# Calculate accuracy
lr_cv_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
lr_cv_eval3.evaluate(lr_cv_predict)
lr_cv_eval3.getMetricName()
#
# Save the lr pipeline model
lr_cv_pipeline_path = "hdfs:///user/rtyle001/CW2/data/lr_cv_pipeline"
lr_cv_pipeline.write().overwrite().save(lr_cv_pipeline_path)
#
# Load the lr pipeline model
lr_cv_pipeline_model = lr_cv_pipeline.load(lr_cv_pipeline_path)
#
# Naive Bayes
nb = NaiveBayes()
nb_paramGrid = ParamGridBuilder()\
.addGrid(nb.smoothing,np.linspace(0.3,10,10))\
.build()
nb_cv = CrossValidator(estimator=nb,\
                       estimatorParamMaps=nb_paramGrid,\
                       evaluator = BinaryClassificationEvaluator(),\
                       numFolds=5)
# Create a pipeline for lr_cv model
nb_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 nb_cv])
nb_cv_model = nb_pipeline.fit(train)
# Transform the pipeline
nb_predict = nb_cv_model.transform(test)
#
# Predictions
nb_predict.printSchema()
nb_select = nb_predict.select("label","prediction","probability").show(5)
nb_predictions.groupBy("label","prediction").count().show()
#
# Evaluate the nb model
# Calculate the area under ROC
nb_cv_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                        metricName="areaUnderROC")
print(nb_cv_eval.evaluate(nb_predict))
print(nb_cv_eval.getMetricName())
#
# Calculate the F1 score
nb_cv_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                             labelCol="label",\
                                             metricName="f1")
print(nb_cv_eval2.evaluate(lr_cv_predict))
print(nb_cv_eval2.getMetricName())
#
# Calculate Accuracy
nb_cv_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                             labelCol="label",\
                                             metricName="accuracy")
print(nb_cv_eval3.evaluate(lr_cv_predict))
print(nb_cv_eval3.getMetricName())
#
# Save the nb pipeline model
nb_pipeline_path = "hdfs:///user/rtyle001/CW2/data/nb_pipeline"
nb_pipeline.write().overwrite().save(nb_pipeline_path)
#
# Load the nb pipeline model
nb_pipeline_model = nb_pipeline.load(nb_pipeline_path)
#
# Decision Tree
# Create decision tree model to data train
dt=DecisionTreeClassifier(featuresCol = 'features',\
                          labelCol = 'label',\
                          maxDepth = 3)
# Create dt pipeline
dt_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 dt])
#
# fit the pipeline model
dt_pipeline = dt_pipeline.fit(train)
# Transform the model on test data
dt_predict=dt_pipeline.transform(test)
# Display the label, prediction and probability
dt_predict.select("label","prediction","probability").show(5)
#
# Confusion Matrix
dt_cm_predict = dt_predict.crosstab("prediction",\
                                   "label").show()
#
# Evaluate the model on test data
# Calulate the area under the curve for the training data 
dt_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(dt_eval.evaluate(dt_predict))
print(dt_eval.getMetricName())
#
# Calculate the f1 score, accuracy, precision and recall using MulticlassClassificationEvaluator
dt_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")

print(dt_eval2.evaluate(dt_predict))
print(dt_eval2.getMetricName())
#
dt_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(dt_eval3.evaluate(nb_predict))
print(dt_eval3.getMetricName())
#
# Save the dt pipeline model
dt_pipeline_path = "hdfs:///user/rtyle001/CW2/data/dt_pipeline"
dt_pipeline.write().overwrite().save(dt_pipeline_path)
#
# Load the lr pipeline model
dt_pipeline_model = dt_pipeline.load(dt_pipeline_path)
#
#Cross Validation
# Create the model
dt_cv=DecisionTreeClassifier(featuresCol="features",\
                            labelCol="label")
#
# Create ParamGrid for Cross Validation
dt_paramGrid = (ParamGridBuilder()\
                .addGrid(dt.maxDepth,[2,5,10])\
                .addGrid(dt.maxBins,[32])\
                .build())
#       
# Create 5-fold CrossValidator
dt_cv = CrossValidator(estimator = dt_cv,\
                       estimatorParamMaps = dt_paramGrid,\
                       evaluator = dt_eval,\
                       numFolds = 5)
# Create dt_cv pipeline
dt_cv_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 dt_cv])
# Run cross validation
dt_cv_model = dt_cv_pipeline.fit(train)
print(dt_cv_model)
# Transform the test data 
dt_cv_predict = dt_cv_model.transform(test)
# Save the dt_cv pipeline model
dt_cv_pipeline.save("hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline")
#
# Load the pipeline model
dt_cv_pipeline = Pipeline.load("hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline")
## Calulate the area under the curve for the training data 
dt_cv_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(dt_cv_eval.evaluate(dt_cv_predict))
print(dt_cv_eval.getMetricName())
#

# Calculate the f1 score, accuracy, precision and recall using MulticlassClassificationEvaluator
dt_cv_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(dt_cv_eval2.evaluate(dt_cv_predict))
print(dt_cv_val2.getMetricName())
#
dt_cv_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(dt_eval3.evaluate(dt_predict))
print(dt_eval3.getMetricName())
#
dt_cv_predict.select("label","prediction").show(5)
dt_cv_predict.groupBy("label","prediction").count().show()
#
# Summary metrics
dt_cv_metrics = MulticlassMetrics(dt_cv_predict)
#
# Save the dt_cv pipeline model
dt_cv_pipeline_path = "hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline"
dt_cv_pipeline.write().overwrite().save(dt_cv_pipeline_path)
#
# Load the lr_cv pipeline model
dt_cv_pipeline_model = dt_pipeline.load(dt_cv_pipeline_path)
#
# Random Forest
# Create the model
rf= RandomForestClassifier(featuresCol="features",\
                           labelCol="label",\
                           numTrees=100,\
                           maxDepth=6)
# Create rf pipeline
rf_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 rf])
#
# Fit the model on train data
rf_pipeline_model=rf_pipeline.fit(train)
# Transform the model on test data
rf_predict=rf_pipeline_model.transform(test)
#
# Display the label, prediction and probability
rf_predict.select("label","prediction","probability").show(5)
# Evaluate the model on test data
rf_eval=BinaryClassificationEvaluator(rawPredictionCol="probability",\
                                     labelCol="label")
#
# Evaluate the rf model
# Calulate the area under the curve
rf_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(rf_eval.evaluate(rf_predict))
print(rf_eval.getMetricName())
#
# Calculate the f1 score
rf_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(rf_eval2.evaluate(rf_predict))
print(rf_eval2.getMetricName())
#
rf_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(rf_eval3.evaluate(rf_predict))
print(rf_eval3.getMetricName())
#
rf_predict.select("label","prediction").show(5)
rf_predict.groupBy("label","prediction").count().show()
#
# Confusion Matrix
rf_cm_predict = rf_predict.crosstab("prediction", "label").show()
#
# Save the rf pipeline model
rf_pipeline_path = "hdfs:///user/rtyle001/CW2/data/rf_pipeline"
rf_pipeline.write().overwrite().save(rf_pipeline_path)
#
# Load the lr_cv pipeline model
rf_pipeline_model = rf_pipeline.load(rf_pipeline_path)
#
# Cross Validation
rf_paramgrid=(ParamGridBuilder()\
              .addGrid(rf.maxDepth,[2,4,6])\
              .addGrid(rf.maxBins,[20,60])\
              .addGrid(rf.numTrees,[5,20])\
              .build())
#
# Creat 5-fold CrossValidator
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_paramgrid, evaluator=rf_eval3, numFolds=5)
#
# Create rf_cv pipeline
rf_cv_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 rf_cv])
#
# Run the CrossValidator
rf_cv_pipeline_model=rf_cv_pipeline.fit(train)
#
# Predictions
rf_cv_predict=rf_cv_pipeline_model.transform(test)
rf_cv_eval.evaluate(rf_cv_predict)
#
# Display some predictions based on occupation
rf_cv_select = rf_cv_predict.select("label", "prediction", "probability", "age", "occupation")
rf_cv_select.show(3)
#
# Evaluate the rf_cv model
# Calulate the area under the curve
rf_cv_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(rf_cv_eval.evaluate(rf_cv_predict))
print(rf_cv_eval.getMetricName())
#
# Calculate the f1 score
rf_cv_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(rf_cv_eval2.evaluate(rf_cv_predict))
print(rf_cv_eval2.getMetricName())
#
# Calculate accuracy
rf_cv_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(rf_cv_eval3.evaluate(rf_cv_predict))
print(rf_cv_eval3.getMetricName())
#
# Save rf_cv pipeline model
rf_cv_pipeline_path = "hdfs:///user/rtyle001/CW2/data/rf_cv_pipeline"
rf_cv_pipeline.write().overwrite().save(rf_cv_pipeline_path)
#
# Load the rf_cv pipeline model
rf_pipeline_model = rf_pipeline.load(rf_pipeline_path)
#
# rf_cv Best model
rf_best_model=rf_cv_pipeline_model.bestModel
rf_final_predict=rf_best_model.transform(test)
rf_cv_eval.evaluate(rf_final_predict)
#
#
# Linear SVM classification
svm = LinearSVC(featuresCol = "features",\
                labelCol = "label",\
                maxIter = 100)
# Create svm pipeline
svm_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 svm])
# Fit the svm model
svm_pipeline_model = svm_pipeline.fit(train)
# Transform the SVM model on test data
svm_predict=svm_pipeline_model.transform(test)
#
# Evaluate the model on test data
svm_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#
# Calulate the area under the curve for the training data 
svm_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(svm_eval.evaluate(svm_predict))
print(svm_eval.getMetricName())
#
# Calculate the f1 score
svm_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(svm_eval2.evaluate(svm_predict))
print(svm_eval2.getMetricName())
#
# Calculate accuracy
svm_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(svm_eval3.evaluate(svm_predict))
print(svm_eval3.getMetricName())
#
# Confusion Matrix
svm_cm_predict = svm_predict.crosstab("prediction","label").show()
#
# Predictions
# Display thelabel, prediction and probability
svm_predict.select("label","prediction","probability").show(5)
svm_predict.groupBy("label","prediction").count().show()
#
# Save the svm pipeline model
svm_pipeline_path = "hdfs:///user/rtyle001/CW2/data/svm_pipeline"
svm_pipeline.write().overwrite().save(svm_pipeline_path)
#
# Load the lr_cv pipeline model
svm_pipeline_model = svm_pipeline.load(svm_pipeline_path)
#
# Gradient Boost model
# Create the model
gbt = GBTClassifier(featuresCol="features",\
                    labelCol="label",\
                    maxIter=10)
#
# Create gbt pipeline
gbt_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 gbt])
# Fit the pipeline model
gbt_pipeline_model = gbt_pipeline.fit(train)
#
# Transfrom the gbt model on test data
gbt_predict = gbt_pipeline_model.transform(test)
#
# Evaluate the gbt model
# Calulate the area under the curve for the training data 
gbt_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(gbt_eval.evaluate(gbt_predict))
print(gbt_eval.getMetricName())
#
# Calculate the f1 score
gbt_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(gbt_eval2.evaluate(gbt_predict))
print(gbt_eval2.getMetricName())
#
# Calculate accuracy
gbt_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(gbt_eval3.evaluate(gbt_predict))
print(gbt_eval3.getMetricName())
#
# Predictions
# Display the label, prediction and probability
gbt_predict.select("label","prediction","probability").show(5)
gbt_predict.groupBy("label","prediction").count().show()
#
# Confusion Matrix
gbt_cm_predict = gbt_predict.crosstab("prediction","label").show()
#
# Save the gbt pipeline model
gbt_pipeline_path = "hdfs:///user/rtyle001/CW2/data/gbt_pipeline"
gbt_pipeline.write().overwrite().save(gbt_pipeline_path)
#
# Load the gbt pipeline model
gbt_pipeline_model = gbt_pipeline.load(gbt_pipeline_path)
#
# Hyperparameter tuning gbt model
print(gbt.explainParams())
gbt_paramGrid = (ParamGridBuilder()\
                 .addGrid(gbt.maxDepth, [2,4,6])\
                 .addGrid(gbt.maxBins, [20,60])\
                 .addGrid(gbt.maxIter,[10,20])\
                 .build())
#
gbt_cv  = CrossValidator(estimator=gbt, estimatorParamMaps = gbt_paramGrid,\
                         evaluator = gbt_eval, numFolds = 5)
# Create gbt_cv pipeline
gbt_cv_pipeline = Pipeline(stages = [stringIndexer,\
                            encoder,\
                            labelToIndex,\
                            assembler,\
                                 gbt_cv])
#
# fit the gbt_cv model
gbt_cv_pipeline_model = gbt_cv_pipeline.fit(train)
#
# Transform the gbt_cv model
gbt_cv_predict = gbt_cv_pipeline_model.transform(test)
#
# Evaluate the gbt_cv model
# Calulate the area under the curve  
gbt__cv_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",\
                                        labelCol="label",\
                                       metricName="areaUnderROC")
print(gbt_cv_eval.evaluate(gbt_cv_predict))
print(gbt_cv_eval.getMetricName())
#
# Calculate the f1 score
gbt_cv_eval2 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                            labelCol="label",\
                                            metricName="f1")
print(gbt_cv_eval2.evaluate(gbt_predict))
print(gbt_cv_eval2.getMetricName())
#
# Calculate accuracy
gbt_cv_eval3 = MulticlassClassificationEvaluator(predictionCol="prediction",\
                                              labelCol="label",\
                                              metricName="accuracy")
print(gbt_cv_eval3.evaluate(gbt_cv_predict))
print(gbt_cv_eval3.getMetricName())
#
# Predictions
# Display the label, prediction and probability
gbt_cv_predict.select("label","prediction","probability").show(5)
gbt_cv_predict.groupBy("label","prediction").count().show()
#
# End of file
sc.stop()