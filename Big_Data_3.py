{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "classification using MLlib\n"
     ]
    }
   ],
      "source": [
          "import pyspark\n",
          "from pyspark.sql import SparkSession\n",
          "from pyspark.sql.types import *\n",
          "from pyspark.sql.functions import isnull, when, count, col\n",
          "from pyspark.sql.functions import mean, avg\n",
          "from pyspark.sql.functions import col, trim, lower\n",
          "from pyspark.sql import Row, SQLContext\n",
          "from pyspark.sql.functions import when, lit\n",
          "from pyspark.sql.functions import *\n",
          "import pyspark.sql.functions as f\n",
          "\n",
          "from pyspark.ml.feature import OneHotEncoder, StringIndexer\n",
          "VectorAssembler, OneHotEncoderModel\n",
          "from pyspark.ml import Pipeline, PipelineModel\n",
          "from pyspark.ml.classification import LogisticRegression\n",
          "GBTClassifier, NaiveBayes, RandomForestClassifier\n",
          "from pyspark.ml.classification import RandomForestClassifier\n",
          "from pyspark.ml.classification import DecisionTreeClassifier\n",
          "from pyspark.ml.classification import LinearSVC\n",
          "from pyspark.ml.tuning import ParamGridBuilder, CrossValidator\n",
          "from pyspark.ml.evaluation import BinaryClassificationEvaluator\n",
          "from pyspark.ml.evaluation import MulticlassClassificationEvaluator\n",
          "from pyspark.ml.stat import Correlation\n",
          "from pyspark.ml.feature import VectorSlicer\n",
          "from pyspark.ml.feature import Bucketizer\n",
          "\n",
          "from pyspark.mllib.evaluation import BinaryClassificationMetrics\n",
          "from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics\n",
          "from pyspark.mllib.util import MLUtils\n",
          "from pyspark.mllib.linalg import Vectors\n",
          "\n",
          "from sklearn.metrics import confusion_matrix\n",
          "from sklearn.metrics import roc_curve, auc\n",
          "from sklearn.metrics import log_loss\n",
          "\n",
          "import pandas as pd\n",
          "import numpy as np\n",
          "import os\n",
          "import sys\n",
          "#Coursework 2: Big Data Analysis March 2022\n",
          "#Income prediction in Pyspark using Ml and MLlib\n",
          "#Predict whether income is <=50k or >50k : the UCI Adult Census dataset \n",
          "# run with spark-submit CW2_jupyter_script.py\n",
          "\n",
          "#Create a SparkSession for appName 'income'\n", 
          "sc = SparkSession.builder.master('local[1]')\
          .appName('income')\
          .getOrCreate()\n",
          "# Load the dataset into a Spark DataFrame\n",
          "df_adult = sc.read.csv('hdfs:///user/rtyle001/CW2/data/adult_data.csv',\
          header=True,\
          inferSchema=True,\
          nanValue='?')\n",
          "\n",
          "# print the dataframe schema\n",
          "df_adult.printSchema()\n",
          "df_adult.show(5, truncate = False)\n",
          "\n",
          "# Drop fnlwgt and 'x' columns\n",
          "df=df_adult.drop('fnlwgt', 'x')\n",
          "df.printSchema()\n",
          "\n",
          "# Cast continuous (numeric) feature columns to FloatTYpe columns\n",
          "columns_to_cast = ['age','capital-gain','educational-num','capital-loss','hours-per-week']\
          df_cast=(df.select(*(c for c in df.columns if c not in columns_to_cast),\
          *(col(c).cast('float').alias(c) for c in columns_to_cast)))\n",
          "df_cast.printSchema()\n",
          "\n",
          "df =df_cast\n",
          "# Exploratory Data Analysis\n",
          "# Summary statistics\n",
          "for col in df.columns:\
              df.describe([col]).show()\n",
          "numeric_features = [t[0] for t in df.dtypes if t[1] == 'float']\n",
          "numeric_data = df.select(numeric_features).toPandas()\n",
          "# Correlation of numeric features\n",
          "numeric_data.corr()\n",
          "\n",
          "# Group By\n",
          "df.groupby('marital-status').agg({'capital-gain': 'mean'}).show()\n",
          "df.groupby('occupation').agg({'capital-gain': 'mean'}).show()\n",
          "df.groupby('income').agg({'educational-num': 'mean'}).show()\n",
          "df.groupby('income').agg({'hours-per-week': 'mean'}).show()\n",
          "df.groupby('race').count().show(truncate=False)\n",
          "df.groupby('workclass').count().show(truncate=False)\n",
          "df.groupby('native-country').count().show(truncate=False)\n",
          "df.groupby('occupation','gender','income').count().show(truncate=False)\n",
          "\n",
          "tot = df.count()\n",
          "df.groupBy('occupation', 'income').count()\
          .withColumnRenamed('count', 'cnt_per_group')\
          .withColumn('perc_of_count_total', (f.col('cnt_per_group') / tot) * 100 )\
          .show()\n",
          "df.groupBy('gender','income').count()\
          .withColumn('%', f.round((f.col('count')/tot)*100,2))\
          .orderBy('count', ascending=False).show()\n",
          "df.groupBy('marital-status','income').count()\
          .withColumn('%', f.round((f.col('count')/tot)*100,2))\
          .orderBy('count', ascending=False).show()\n",
          "df.groupBy('race','income').count() \
          .withColumn('%', f.round((f.col('count')/tot)*100,2)) \
          .orderBy('count', ascending=False).show()\n",
          "df.groupBy('education','income').count()\
          .withColumn('%', f.round((f.col('count')/tot)*100,2))\
          .orderBy('count', ascending=False).show()\n",
          "df.groupBy('native-country').count() \
          .withColumn('%', f.round((f.col('count')/tot)*100,2)) \
          .orderBy('count', ascending=False) \
          .show()\n",
          "\n",
          "# print count of row and columns\n",
          "print('There are {} columns and {} rows in the adult dataset.'\
          .format(len(df.columns), df.count()))\n",
          "# Missing data\n",
          "# Count the value of '?' in every column\n",
          "for col in df.columns:\
              print(col, '\t', 'with '?' values: ',\
                  df.filter(df[col]=='?').count())\n",
          "# Replace entries with '?' with 'None'\n",
          "df2=df.replace('?',None)\n",
          "# Do a validation Count the value of '?' in every column\n",
          "for col in df2.columns:\
              print(col, '\t', 'with '?' values: ',df2.filter(df2[col]=='?').count())\n",
          "# Count the number of null values in every column\n",
          "for col in df2.columns:\
              print(col, '\t', 'with null values: ',\
                  df2.filter(df2[col].isNull()).count())\n",
          "# Drop null values\n",
          "df_no_null =df2.na.drop()\n",
          "\n",
          "# Validate there are no nulls\n",
          "for col in df_no_null.columns:\
              print(col, '\t', 'with null values: ',\
                  df_no_null.filter(df_no_null[col].isNull())\
          .count())\n",
          "\n",
          "# Bucketizer\n",
          "# Group by age\n",
          "df.groupby('age').count()\
          .orderBy('age',ascending=True)\
          .show()\n",
          "df.agg({'age':'min'}).show()\n",
          "df.agg({'age':'max'}).show()\n",
          "splits = [17, 30, 50, 70, 90]\n",
          "bucketizer  = Bucketizer( splits=splits,inputCol='age', outputCol='age_group')\n",
          "df_age = bucketizer.transform(df)\n",
          "df_age.select('age_group').show()\n",
          "\n",
          "# Create a new feature for 'age' column - add a square to age\n",
          "from pyspark.sql.functions import col, trim, lower\n",
          "df_calc = df_no_null.withColumn('age-squared',col('age')**2)\n",
          "df_calc.printSchema()\n",
          "\n",
          "# Count and sort by country\n",
          "df_calc.groupby('native-country')\
          .agg({'native-country':'count'})\
          .sort(asc('count(native-country)'))\
          .show()\n",
          "# Holand-Netherlands only has 1 record\n",
          "# Filter it out of the dataframe\n",
          "df3=df_calc.filter(df_calc['native-country']!='Holand-Netherlands')\n",
          "\n",
          "# compute count of instances per label: class 0: <= 50k income; class 1: > 50k income\n",
          "class_counts = df3.groupBy('income')\
          .count().collect()\n",
          "\n",
          "# add the counts to a dictionary for display\n",
          "class_counts_dict = {row['income']: row['count'] for row in class_counts}\n",
          "class_counts_dict\n",
          "\n",
          "# print count of row and columns\n",
          "print('There are {} columns and {} rows in the adult dataset.'\
          .format(len(df3.columns), df3.count()))\n",
          "# there are 15 columns and 45221 rows\n",
          "\n",
          "# Split the data into training and testing datasets\n",
          "# Split 80/20\n",
          "train, test = df3.randomSplit([.8,.2], seed=13)\n",
          "\n",
          "# Category Indexing with StringIndexer\n",
          "# Categorical columns\n",
          "cat_cols = ['workclass', 'education', 'marital-status',\
          'occupation', 'relationship', 'race', 'gender', 'native-country']\n",
          "# Numerical columns\n",
          "num_cols=['age','capital-gain','educational-num',\
          'capital-loss','hours-per-week','age-squared']\n",
          "# String Indexer\n",
          "stringIndexer = StringIndexer(inputCols=cat_cols,\
          outputCols=[x + 'Index' for x in cat_cols])\
          .setHandleInvalid('skip')\n",
          "encoder = OneHotEncoder(inputCols=stringIndexer\
          .getOutputCols(),outputCols=[x + 'OHE' for x in cat_cols])\n",
          "# The label column ('income') is also a string value \
          - it has two possible values, '<=50K' and '>50K'\n",
          "# Convert it to a numeric value using StringIndexer.\n",
          "labelToIndex = StringIndexer(inputCol='income',\
          'outputCol='label')\
          .setHandleInvalid('skip')\n",
          "stringIndexerModel = stringIndexer.fit(df3)\n",
          "stringIndexerModel.transform(df3).show(5)\n",
          "\n",
          "# Vector Assembler\n",
          "assemblerInputs = [c + 'OHE' for c in cat_cols] + num_cols\n",
          "assembler= VectorAssembler(inputCols=assemblerInputs,\
          outputCol='features')\n",
          "\n",
          "# Logistic Regression Model\n",
          "# Create the lr model\n",
          "lr = LogisticRegression(featuresCol='features,labelCol='label', maxIter=10)\n",
          "\n",
          "# Create a pipeline for lr model\n",
          "lr_pipeline= Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          lr])\n",
          "# fit the lr model to training data\n",
          "lr_pipeline = lr_pipeline.fit(train)\n",
          "\n",
          "# transform the lr model to test data\n",
          "lr_predict = lr_pipeline.transform(test)\n",
          "\n",
          "# PrintSchema of lr and mlr predictions\n",
          "lr_predict.printSchema()\n",
          "\n",
          "# Display the predictions for lr and mlr models\n",
          "lr_select = lr_predict.select('label', 'prediction', 'probability').show(5)\n",
          "# Confusion Matrix\n",
          "lr_cm_predict = lr_predict.crosstab('prediction', 'label').show()\n",
          "\n",
          "# Evaluate the lr model\n",
          "# Calulate the area under the curve \n",
          "lr_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(lr_eval.evaluate(lr_predict))\n",
          "print(lr_eval.getMetricName())\n",
          "\n",
          "# Calculate the f1 score\n",
          "lr_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "print(lr_eval2.evaluate(lr_predict))\n",
          "print(lr_eval2.getMetricName())\n",
          "\n",
          "# Calculate accuracy\n",
          "lr_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(lr_eval3.evaluate(lr_predict))\n",
          "print(lr_eval3.getMetricName())\n",
          "\n",
          "# Save the lr pipeline model\n",
          "lr_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/lr_pipeline'\n",
          "lr_pipeline.write().overwrite().save(lr_pipeline_path)\n",
          "\n",
          "# Load the lr pipeline model\n",
          "lr_pipeline_model = lr_pipeline.load(lr_pipeline_path)\n",
          "\n",
          "# Hyper-Parameter tuning: Logistic Regression\n",
          "# Create the model\n",
          "lr_cv=LogisticRegression(featuresCol='features',\
          labelCol='label')\n",
          "\n",
          "# Create ParamGrid for cross validataion\n",
          "lr_cv_paramGrid=ParamGridBuilder()\
          .addGrid(lr.regParam,[0.1, 0.5, 2.0])\
          .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
          .addGrid(lr.maxIter,[1,5,10])\
          .build()\n",
          "\n",
          "lr_cv=CrossValidator(estimator=lr_cv,\
          estimatorParamMaps=lr_cv_paramGrid,\
          evaluator=BinaryClassificationEvaluator(),\
          numFolds=3)\n",
          "\n",
          "# Create a pipeline for lr_cv model\n",
          "lr_cv_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          lr_cv])\n",
          "\n",
          "# fit the lr model to training data\n",
          "lr_cv_pipeline = lr_cv_pipeline.fit(train)\n",
          "\n",
          "# transform the lr model to test data\n",
          "lr_cv_predict = lr_cv_pipeline.transform(test)\n",
          "\n",
          "# Predictions\n",
          "lr_cv_predict.printSchema()\n",
          "lr_cv_select = lr_cv_predict.select('label','prediction','probability').show(5)\n",
          "\n",
          "# Evaluate the lr_cv model\n",
          "# Calculate area under curve\n",
          "lr_cv_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(lr_cv_eval.evaluate(lr_cv_predict))\n",
          "print(lr_cv_eval.getMetricName())\n",
          "\n",
          "# Calculate the f1 score\n",
          "lr_cv_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "lr_cv_eval2.evaluate(lr_cv_predict)\n",
          "lr_cv_eval2.getMetricName()\n",
          "\n",
          "# Calculate accuracy\n",
          "lr_cv_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "lr_cv_eval3.evaluate(lr_cv_predict)\n",
          "lr_cv_eval3.getMetricName()\n",
          "\n",
          "# Save the lr pipeline model\n",
          "lr_cv_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/lr_cv_pipeline'\n",
          "lr_cv_pipeline.write().overwrite().save(lr_cv_pipeline_path)\n",
          "\n",
          "# Load the lr pipeline model\n",
          "lr_cv_pipeline_model = lr_cv_pipeline.load(lr_cv_pipeline_path)\n",
          "\n",
          "# Naive Bayes\n",
          "nb = NaiveBayes()\n",
          "nb_paramGrid = ParamGridBuilder()\
          .addGrid(nb.smoothing,np.linspace(0.3,10,10))\
          .build()\n",
          "nb_cv = CrossValidator(estimator=nb,\
          estimatorParamMaps=nb_paramGrid,\
          evaluator = BinaryClassificationEvaluator(),\
          numFolds=5)\n",
          "# Create a pipeline for lr_cv model\n",
          "nb_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          nb_cv])\n",
          "nb_cv_model = nb_pipeline.fit(train)\n",
          "# Transform the pipeline\n",
          "nb_predict = nb_cv_model.transform(test)\n",
          "\n",
          "# Predictions\n",
          "nb_predict.printSchema()\n",
          "nb_select = nb_predict.select('label','prediction','probability').show(5)\n",
          "nb_predictions.groupBy('label','prediction').count().show()\n",
          "\n",
          "# Evaluate the nb model\n",
          "# Calculate the area under ROC\n",
          "nb_cv_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(nb_cv_eval.evaluate(nb_predict))\n",
          "print(nb_cv_eval.getMetricName())\n",
          "\n",
          "# Calculate the F1 score\n",
          "nb_cv_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "print(nb_cv_eval2.evaluate(lr_cv_predict))\n",
          "print(nb_cv_eval2.getMetricName())\n",
          "\n",
          "# Calculate Accuracy\n",
          "nb_cv_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(nb_cv_eval3.evaluate(lr_cv_predict))\n",
          "print(nb_cv_eval3.getMetricName())\n",
          "\n",
          "# Save the nb pipeline model\n",
          "nb_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/nb_pipeline'\n",
          "nb_pipeline.write().overwrite().save(nb_pipeline_path)\n",
          "\n",
          "# Load the nb pipeline model\n",
          "nb_pipeline_model = nb_pipeline.load(nb_pipeline_path)\n",
          "\n",
          "# Decision Tree\n",
          "# Create decision tree model to data train\n",
          "dt=DecisionTreeClassifier(featuresCol = 'features',\
          labelCol = 'label',\
          maxDepth = 3)\n",
          "# Create dt pipeline\n",
          "dt_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          dt])\n",
          "\n",
          "# fit the pipeline model\n",
          "dt_pipeline = dt_pipeline.fit(train)\n",
          "# Transform the model on test data\n",
          "dt_predict=dt_pipeline.transform(test)\n",
          "# Display the label, prediction and probability\n",
          "dt_predict.select('label','prediction','probability').show(5)\n",
          "\n",
          "# Confusion Matrix\n",
          "dt_cm_predict = dt_predict.crosstab('prediction',\
          'label').show()\n",
          "\n",
          "# Evaluate the model on test data\n",
          "# Calulate the area under the curve for the training data\n",
          "dt_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(dt_eval.evaluate(dt_predict))\n",
          "print(dt_eval.getMetricName())\n",
          "\n",
          "# Calculate the f1 score\n",
          "dt_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "\n",
          "print(dt_eval2.evaluate(dt_predict))\n",
          "print(dt_eval2.getMetricName())\n",
          "\n",
          "# Calculate accuracy\n",
          "dt_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(dt_eval3.evaluate(nb_predict))\n",
          "print(dt_eval3.getMetricName())\n",
          "\n",
          "# Save the dt pipeline model\n",
          "dt_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/dt_pipeline'\n",
          "dt_pipeline.write().overwrite().save(dt_pipeline_path)\n",
          "\n",
          "# Load the lr pipeline model\n",
          "dt_pipeline_model = dt_pipeline.load(dt_pipeline_path)\n",
          "\n",
          "#Cross Validation\n",
          "# Create the model\n",
          "#dt_cv=DecisionTreeClassifier(featuresCol='features',\
          labelCol='label')\n",
          "# Create ParamGrid for Cross Validation\n",
          "#dt_paramGrid = (ParamGridBuilder()\
          .addGrid(dt.maxDepth,[2,5,10])\
          .addGrid(dt.maxBins,[32])\
          .build())\n",
          "# Create 5-fold CrossValidator\n",
          "#dt_cv = CrossValidator(estimator = dt_cv,\
          estimatorParamMaps = dt_paramGrid,\
          evaluator = dt_eval,\
          numFolds = 5)\n",
          "# Create dt_cv pipeline\n",
          "#dt_cv_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          dt_cv])\n",
          "# Run cross validation\n",
          "#dt_cv_model = dt_cv_pipeline.fit(train)\n",
          "#print(dt_cv_model)\n",
          "# Transform the test data\n",
          "#dt_cv_predict = dt_cv_model.transform(test)\n",
          "# Save the dt_cv pipeline model\n",
          "#dt_cv_pipeline.save('hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline')\n",
          "# Load the pipeline model\n",
          "#dt_cv_pipeline = Pipeline.load('hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline')\n",
          "# Calulate the area under the curve for the training data\n",
          "#dt_cv_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "#print(dt_cv_eval.evaluate(dt_cv_predict))\n",
          "#print(dt_cv_eval.getMetricName())\n",
          "# Calculate the f1 score\n",
          "#dt_cv_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "#print(dt_cv_eval2.evaluate(dt_cv_predict))\n",
          "#print(dt_cv_val2.getMetricName())\n",
          "# Calculate accuracy\n",
          "#dt_cv_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "#print(dt_eval3.evaluate(dt_predict))\n",
          "#print(dt_eval3.getMetricName())\n",
          "#dt_cv_predict.select('label','prediction').show(5)\n",
          "#dt_cv_predict.groupBy('label','prediction').count().show()\n",
          "# Summary metrics\n",
          "#dt_cv_metrics = MulticlassMetrics(dt_cv_predict)\n",
          "# Save the dt_cv pipeline model\n",
          "#dt_cv_pipeline_path = hdfs:///user/rtyle001/CW2/data/dt_cv_pipeline'\n",
          "#dt_cv_pipeline.write().overwrite().save(dt_cv_pipeline_path)\n",
          "# Load the lr_cv pipeline model\n",
          "#dt_cv_pipeline_model = dt_pipeline.load(dt_cv_pipeline_path)\n",
          "# Random Forest\n",
          "# Create the model\n",
          "rf= RandomForestClassifier(featuresCol='features',\
          labelCol='label',\
          numTrees=100,\
          maxDepth=6)\n",
          "# Create rf pipeline\n",
          "rf_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          rf])\n",
          "\n",
          "# Fit the model on train data\n",
          "rf_pipeline_model=rf_pipeline.fit(train)\n",
          "# Transform the model on test data\n",
          "rf_predict=rf_pipeline_model.transform(test)\n",
          "\n",
          "# Display the label, prediction and probability\n",
          "rf_predict.select('label','prediction','probability').show(5)\n",
          "# Evaluate the model on test data\n",
          "rf_eval=BinaryClassificationEvaluator(rawPredictionCol='probability',\
          labelCol='label')\n",
          "\n",
          "# Evaluate the rf model\n",
          "# Calulate the area under the curve\n",
          "rf_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(rf_eval.evaluate(rf_predict))\n",
          "print(rf_eval.getMetricName())\n",
          "\n",
          "# Calculate the f1 score\n",
          "rf_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "print(rf_eval2.evaluate(rf_predict))\n",
          "print(rf_eval2.getMetricName())\n",
          "\n",
          "rf_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(rf_eval3.evaluate(rf_predict))\n",
          "print(rf_eval3.getMetricName())\n",
          "\n",
          "rf_predict.select('label','prediction').show(5)\n",
          "rf_predict.groupBy('label','prediction').count().show()\n",
          "\n",
          "# Confusion Matrix\n",
          "rf_cm_predict = rf_predict.crosstab('prediction', 'label').show()\n",
          "\n",
          "# Save the rf pipeline model\n",
          "rf_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/rf_pipeline'\n",
          "rf_pipeline.write().overwrite().save(rf_pipeline_path)\n",
          "\n",
          "# Load the lr_cv pipeline model\n",
          "rf_pipeline_model = rf_pipeline.load(rf_pipeline_path)\n",
          "\n",
          "# Cross Validation\n",
          "#rf_paramgrid=(ParamGridBuilder()\
          .addGrid(rf.maxDepth,[2,4,6])\
          .addGrid(rf.maxBins,[20,60])\
          .addGrid(rf.numTrees,[5,20])\
          .build())\n",
          "# Create 5-fold CrossValidator\n",
          "#rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_paramgrid, evaluator=rf_eval3, numFolds=5)\n",
          "# Create rf_cv pipeline\n",
          "#rf_cv_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          rf_cv])\n",
          "\n",
          "# Run the CrossValidator\n",
          "#rf_cv_pipeline_model=rf_cv_pipeline.fit(train)\n",
          "# Predictions\n",
          "#rf_cv_predict=rf_cv_pipeline_model.transform(test)\n",
          "#rf_cv_eval.evaluate(rf_cv_predict)\n",
          "# Display some predictions based on occupation\n",
          "#rf_cv_select = rf_cv_predict.select('label', 'prediction', 'probability', 'age', 'occupation')\n",
          "#rf_cv_select.show(3)\n",
          "\n",
          "# Evaluate the rf_cv model\n",
          "# Calulate the area under the curve\n",
          "#rf_cv_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "#print(rf_cv_eval.evaluate(rf_cv_predict))\n",
          "#print(rf_cv_eval.getMetricName())\n",
          "# Calculate the f1 score\n",
          "#rf_cv_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "#print(rf_cv_eval2.evaluate(rf_cv_predict))\n",
          "#print(rf_cv_eval2.getMetricName())\n",
          "# Calculate accuracy\n",
          "#rf_cv_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "#print(rf_cv_eval3.evaluate(rf_cv_predict))\n",
          "#print(rf_cv_eval3.getMetricName())\n",
          "# Save rf_cv pipeline model\n",
          "#rf_cv_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/rf_cv_pipeline'\n",
          "#rf_cv_pipeline.write().overwrite().save(rf_cv_pipeline_path)\n",
          "# Load the rf_cv pipeline model\n",
          "#rf_pipeline_model = rf_pipeline.load(rf_pipeline_path)\n",
          "\n",
          "# Linear SVM classification\n",
          "svm = LinearSVC(featuresCol = 'features',\
          labelCol = 'label',\
          maxIter = 100)\n",
          "# Create svm pipeline\n",
          "svm_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          svm])\n",
          "# Fit the svm model\n",
          "svm_pipeline_model = svm_pipeline.fit(train)\n",
          "# Transform the SVM model on test data\n",
          "svm_predict=svm_pipeline_model.transform(test)\n",
          "# Evaluate the model on test data\n",
          "svm_eval = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')\n",
          "# Calulate the area under the curve for the training data\n",
          "svm_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(svm_eval.evaluate(svm_predict))\n",
          "print(svm_eval.getMetricName())\n",
          "# Calculate the f1 score\n",
          "svm_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "print(svm_eval2.evaluate(svm_predict))\n",
          "print(svm_eval2.getMetricName())\n",
          "# Calculate accuracy\n",
          "svm_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(svm_eval3.evaluate(svm_predict))\n",
          "print(svm_eval3.getMetricName())\n",
          "# Confusion Matrix\n",
          "svm_cm_predict = svm_predict.crosstab('prediction','label').show()\n",
          "# Predictions\n",
          "# Display thelabel, prediction and probability\n",
          "svm_predict.select('label','prediction','probability').show(5)\n",
          "svm_predict.groupBy('label','prediction').count().show()\n",
          "# Save the svm pipeline model\n",
          "svm_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/svm_pipeline\n",
          "svm_pipeline.write().overwrite().save(svm_pipeline_path)\n",
          "# Load the lr_cv pipeline model\n",
          "svm_pipeline_model = svm_pipeline.load(svm_pipeline_path)\n",
          "# Gradient Boost model\n",
          "# Create the model\n",
          "gbt = GBTClassifier(featuresCol='features',\
          labelCol='label',\
          maxIter=10)\n",
          "# Create gbt pipeline\n",
          "gbt_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          gbt])\n",
          "# Fit the pipeline model\n",
          "gbt_pipeline_model = gbt_pipeline.fit(train)\n",
          "# Transfrom the gbt model on test data\n",
          "gbt_predict = gbt_pipeline_model.transform(test)\n",
          "# Evaluate the gbt model\n",
          "# Calulate the area under the curve for the training data\n",
          "gbt_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "print(gbt_eval.evaluate(gbt_predict))\n",
          "print(gbt_eval.getMetricName())\n",
          "# Calculate the f1 score\n",
          "gbt_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "print(gbt_eval2.evaluate(gbt_predict))\n",
          "print(gbt_eval2.getMetricName())\n",
          "# Calculate accuracy\n",
          "gbt_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "print(gbt_eval3.evaluate(gbt_predict))\n",
          "print(gbt_eval3.getMetricName())\n",
          "# Predictions\n",
          "# Display the label, prediction and probability\n",
          "gbt_predict.select('label','prediction','probability').show(5)\n",
          "gbt_predict.groupBy('label','prediction').count().show()\n",
          "# Confusion Matrix\n",
          "gbt_cm_predict = gbt_predict.crosstab('prediction','label').show()\n",
          "# Save the gbt pipeline model\n",
          "gbt_pipeline_path = 'hdfs:///user/rtyle001/CW2/data/gbt_pipeline'\n",
          "gbt_pipeline.write().overwrite().save(gbt_pipeline_path)\n",
          "# Load the gbt pipeline model\n",
          "gbt_pipeline_model = gbt_pipeline.load(gbt_pipeline_path)\n",
          "# Hyperparameter tuning gbt model\n",
          "#print(gbt.explainParams())\n",
          "#gbt_paramGrid = (ParamGridBuilder()\
          .addGrid(gbt.maxDepth, [2,4,6])\
          .addGrid(gbt.maxBins, [20,60])\
          .addGrid(gbt.maxIter,[10,20])\
          .build())\n",
          "#gbt_cv  = CrossValidator(estimator=gbt, estimatorParamMaps = gbt_paramGrid,\
          evaluator = gbt_eval, numFolds = 5)\n",
          "# Create gbt_cv pipeline\n",
          "#gbt_cv_pipeline = Pipeline(stages = [stringIndexer,\
          encoder,\
          labelToIndex,\
          assembler,\
          gbt_cv])\n",
          "# fit the gbt_cv model\n",
          "gbt_cv_pipeline_model = gbt_cv_pipeline.fit(train)\n",
          "# Transform the gbt_cv model\n",
          "#gbt_cv_predict = gbt_cv_pipeline_model.transform(test)\n",
          "# Evaluate the gbt_cv model\n",
          "# Calulate the area under the curve\n",
          "#gbt__cv_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',\
          labelCol='label',\
          metricName='areaUnderROC')\n",
          "#print(gbt_cv_eval.evaluate(gbt_cv_predict))\n",
          "#print(gbt_cv_eval.getMetricName())\n",
          "# Calculate the f1 score\n",
          "#gbt_cv_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='f1')\n",
          "#print(gbt_cv_eval2.evaluate(gbt_predict))\n",
          "#print(gbt_cv_eval2.getMetricName())\n",
          "# Calculate accuracy\n",
          "#gbt_cv_eval3 = MulticlassClassificationEvaluator(predictionCol='prediction',\
          labelCol='label',\
          metricName='accuracy')\n",
          "#print(gbt_cv_eval3.evaluate(gbt_cv_predict))\n",
          "#print(gbt_cv_eval3.getMetricName())\n",
          "# Predictions\n",
          "# Display the label, prediction and probability\n",
          "#gbt_cv_predict.select('label','prediction','probability').show(5)\n",
          "#gbt_cv_predict.groupBy('label','prediction').count().show()\n",
          "# End of file\n",
          "sc.stop()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "PySpark",
   "language": "python",
   "name": "pyspark"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "classification/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}