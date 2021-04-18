import os
import warnings
import sys

import pyspark
from pyspark import SparkFiles
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np
import mlflow
import mlflow.spark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

if __name__ == "__main__":

    sc = SparkContext('local')
    spark = SparkSession(sc)
    spark.sparkContext.addFile("https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv")

    try:
        los_sparkDF = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("file://"+SparkFiles.get("LengthOfStay.csv"))
        los_sparkDF.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/lengthofstay")
        spark.sql("DROP TABLE IF EXISTS lengthofstay")
        spark.sql("CREATE TABLE lengthofstay USING DELTA LOCATION '/mnt/delta/lengthofstay/'")
        train_df = spark.sql("SELECT * FROM lengthofstay")
    except Exception as e:
        print("Unable to read training & test data from Delta Lake: {0}".format(e))

    categoricalColumns = ["gender","rcount","facid"]
    stages = [] # stages in our Pipeline
    for categoricalCol in categoricalColumns:
        # Category Indexing with StringIndexer
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
        # Use OneHotEncoder to convert categorical variables into binary SparseVectors
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        # Add stages.  These are not run here, but will run all at once later on.
        stages += [stringIndexer, encoder]
        
    # Transform all features into a vector using VectorAssembler
    numericCols = ['eid', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo', 'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9']
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="rawFeatures")
    stages += [assembler]

    # vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
    vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=6)
    stages += [vectorIndexer]

    (trainingData, testData) = train_df.randomSplit([0.7, 0.3], seed=100)

    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a GBTRegressor model.
    gbt = GBTRegressor(labelCol="lengthofstay")

    # Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
    evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

    singleModelstages = stages + [gbt]
    singleModelpipeline = Pipeline().setStages(singleModelstages)

    with mlflow.start_run(run_name='los-experiment') as run:
        modelPipeline = singleModelpipeline.fit(trainingData)
        
        mlflow.log_metric('Mean squared error', evaluator.evaluate(modelPipeline.transform(testData)))
        print(evaluator.evaluate(modelPipeline.transform(testData)))

        # Log the best model.
        mlflow.spark.log_model(modelPipeline, artifact_path="spark-model", registered_model_name="mlflowproj_LengthOfStaySparkModel")