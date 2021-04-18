# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Healthcare Demo
# MAGIC ## Chapter 3: Classical ML & Patient Length of Stay

# COMMAND ----------

# MAGIC %md
# MAGIC ![image info](https://visualassets.blob.core.windows.net/flow-diagrams/classic-ml.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC With SQL Analytics, we were able to identify several contributing factors to help us better forecast patient length of stay. With an improved understanding of length of stay, Contoso Hospital can optimize bed usage and maximize treatment availability.
# MAGIC 
# MAGIC Let's see how classical machine learning techniques can help us use our new data and insight to predict future bed availability.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Single model training
# MAGIC 
# MAGIC We'll use PySpark ML to start with and manually train a model.
# MAGIC 
# MAGIC We'll also use Databrick's MLflow support to manage this end-to-end flow. It tracks our training experiments, manages models, and serves models as REST endpoints.

# COMMAND ----------

# DBTITLE 1,Fetch combined data from Delta Lake 
train_df = spark.sql("SELECT * FROM lengthofstay")
display(train_df)

# COMMAND ----------

# DBTITLE 1,Encode data for training
import pyspark
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, VectorIndexer

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

display(train_df)

# COMMAND ----------

# DBTITLE 1,Set MLflow tracking URI to local
import mlflow

mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Train model with PySpark & MLflow
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np
import mlflow
import mlflow.spark

(trainingData, testData) = train_df.randomSplit([0.7, 0.3], seed=100)

# The next step is to define the model training stage of the pipeline. 
# The following command defines a GBTRegressor model.
gbt = GBTRegressor(labelCol="lengthofstay")

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

singleModelstages = stages + [gbt]
singleModelpipeline = Pipeline().setStages(singleModelstages)

with mlflow.start_run(run_name='los-experiment-spark') as run:
  modelPipeline = singleModelpipeline.fit(trainingData)
  
  mlflow.log_metric('Mean squared error', evaluator.evaluate(modelPipeline.transform(testData)))
  
  # Log the best model.
  mlflow.spark.log_model(modelPipeline, artifact_path="spark-model", registered_model_name="LengthOfStaySparkModel") 

# COMMAND ----------

# DBTITLE 1,Evaluate model
predictions = modelPipeline.transform(testData)
rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

# DBTITLE 1,Wait for model to be registered
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(1000):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    if status == ModelVersionStatus.READY:
      print("Model status: %s" % ModelVersionStatus.to_string(status))
      break
    time.sleep(1)

client = MlflowClient()
model_version_infos = client.search_model_versions("name = '%s'" % "LengthOfStaySparkModel")
latest_model_version = max([model_version_info.version for model_version_info in model_version_infos], key=int)

wait_until_ready("LengthOfStaySparkModel", latest_model_version)

# COMMAND ----------

# DBTITLE 1,Promote model to production
output = client.transition_model_version_stage(
  name="LengthOfStaySparkModel",
  version=int(latest_model_version),
  stage="Production",
)

print(output.current_stage, " -> ", output.status)

# COMMAND ----------

# DBTITLE 1,Prepare to query production model
import os

os.environ["DATABRICKS_TOKEN"] = "dapi4523eb5cd177a1440a090e13bcb2dbfc"

# 
#
# Copy the below code from the Models tab after Model Serving has been enabled.
#
#
import os
import requests
import pandas as pd

def score_model(dataset: pd.DataFrame):
  url = 'https://adb-6739797150782991.11.azuredatabricks.net/model/LengthOfStaySparkModel/Production/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# DBTITLE 1,Wait until model is ready
import requests

url = 'https://adb-6739797150782991.11.azuredatabricks.net/model/LengthOfStaySparkModel/Production/invocations'

# Wait until the model serving is ready
def wait_until_url_ready(url):
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  for _ in range(1000):
    r = requests.request(method='POST', headers=headers, url=url)
    if r.status_code == 415:
      print("Model serving: READY")
      break
    time.sleep(1)
    
wait_until_url_ready(url)

# COMMAND ----------

# DBTITLE 1,Real-time model scoring
# Model serving is designed for low-latency predictions on smaller batches of data
num_predictions = 5
served_predictions = score_model(testData.toPandas()[:num_predictions])
model_evaluations = list(modelPipeline.transform(testData.limit(num_predictions)).select('prediction').toPandas()['prediction'])
# Compare the results from the deployed model and the trained model
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions,
})

# COMMAND ----------

# DBTITLE 1,Batch model scoring
from pyspark.sql.functions import struct
import mlflow.pyfunc

testData.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/lengthofstaybatch")
 
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/LengthOfStaySparkModel/production")

# Read the "new data" from Delta
new_data = spark.read.format("delta").load("/mnt/delta/lengthofstaybatch")
 
# Apply the model to the new data
udf_inputs = struct(*(testData.columns))
 
new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

display(new_data.select("eid","lengthofstay","prediction"))

# COMMAND ----------

# DBTITLE 1,Write predictions to Delta Lake
new_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/predictions")
spark.sql("DROP TABLE IF EXISTS predictions")
spark.sql("CREATE TABLE predictions USING DELTA LOCATION '/mnt/delta/predictions/'")
new_data = spark.sql("SELECT * FROM predictions")
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Multiple model training
# MAGIC 
# MAGIC Now that we have a model in production, let's see how we can improve its performance through hyperparameter tuning.
# MAGIC 
# MAGIC We'll use MLflow and PySpark to explore a grid of hyperparameters to train an optimal gradient-boosted decision tree model. 

# COMMAND ----------

# DBTITLE 1,Define GBTRegressor with hyperparameter training stages
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np

# The next step is to define the model training stage of the pipeline. 
# The following command defines a GBTRegressor model. 
gbt = GBTRegressor(labelCol="lengthofstay")
 
# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 200])\
  .build()
 
# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
 
# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

hyperparamModelstages = stages + [cv]
hyperparamModelpipeline = Pipeline().setStages(hyperparamModelstages)

# COMMAND ----------

# DBTITLE 1,Train model with PySpark & MLflow
import mlflow
import mlflow.spark

with mlflow.start_run(run_name='los-experiment-hyperparam'):
  modelPipeline = hyperparamModelpipeline.fit(trainingData)
  
  test_metric = evaluator.evaluate(modelPipeline.transform(testData))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric) 
  
  # Log the best model.
  mlflow.spark.log_model(modelPipeline, artifact_path="spark-model", registered_model_name="LengthOfStaySparkModel") 

# COMMAND ----------

# DBTITLE 1,Evaluate model
predictions = modelPipeline.transform(testData)

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

# DBTITLE 1,Wait for model to be registered
from mlflow.tracking import MlflowClient
 
client = MlflowClient()
model_version_infos = client.search_model_versions("name = '%s'" % "LengthOfStaySparkModel")
latest_model_version = max([model_version_info.version for model_version_info in model_version_infos], key=int)
wait_until_ready("LengthOfStaySparkModel", latest_model_version)

# COMMAND ----------

# DBTITLE 1,Promote new model to staging
output = client.transition_model_version_stage(
  name="LengthOfStaySparkModel",
  version=int(latest_model_version),
  stage="Staging",
)

print(output.current_stage, " -> ", output.status)

# COMMAND ----------

# DBTITLE 1,Batch model scoring
from pyspark.sql.functions import struct
import mlflow.pyfunc

testData.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/lengthofstaybatch")
 
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/LengthOfStaySparkModel/staging")
# Read the "new data" from Delta
new_data = spark.read.format("delta").load("/mnt/delta/lengthofstaybatch")
 
# Apply the model to the new data
udf_inputs = struct(*(testData.columns))
 
new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

display(new_data.select("eid","lengthofstay","prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Integration
# MAGIC 
# MAGIC MLflow also provides first-class support for Azure ML. By updating the tracking URI, the experiment information captured by MLflow can be recorded as a native Azure ML experiment.
# MAGIC 
# MAGIC Let's build a sklearn model and see how Azure ML can help us manage the end-to-end experiment lifecycle.

# COMMAND ----------

# DBTITLE 1,Integrate MLflow with Azure ML
import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core
from azureml.core.authentication import InteractiveLoginAuthentication

from azureml.core import Workspace

try:    
    interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
    # Get instance of the Workspace and write it to config file
    ws = Workspace(
        subscription_id = '2a779d6f-0806-4359-a6e8-f1fd57bb5dd7', 
        resource_group = 'azure-databricks-demo', 
        workspace_name = 'databricks-demo-ml-ws',
        auth = interactive_auth)
    # Writes workspace config file
    ws.write_config()
    
    print('Library configuration succeeded')
except Exception as e:
    print(e)
    print('Workspace not found')

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

# DBTITLE 1,Train model with sklearn & MLflow
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import ensemble

X = train_df.toPandas()
y = X['lengthofstay']
X = X.loc[:, X.columns != 'lengthofstay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.1,
          'loss': 'ls'}

column_tranformer = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore'), ["gender","rcount","facid"])])

clf = Pipeline(steps=[('preprocessor', column_tranformer),
                      ('classifier',  ensemble.GradientBoostingRegressor(**params))])


experimentName = "patient-los-sklearn" 
mlflow.set_experiment(experimentName) 

with mlflow.start_run():
  # Train the model using the training sets
  clf.fit(X_train, y_train)

  # Make predictions using the testing set
  preds = clf.predict(X_test)

  mlflow.log_metric('Mean squared error', mean_squared_error(y_test, preds, squared=False))
  
  mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name="LengthOfStaySklearnModel")

# COMMAND ----------

# DBTITLE 1,Evaluate model
print("RMSE on our test set: %g" % mean_squared_error(y_test, preds, squared=False))

# COMMAND ----------

# MAGIC %md
# MAGIC With Azure ML, we gain access to hosting options like Azure Kubernetes Service for our model. Let's see how we can provision a k8s cluster and deploy our model, all from Databricks.

# COMMAND ----------

# DBTITLE 1,Create AKS cluster
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()

aks_name = 'aks-mlflow'

try:
  aks_target = AksCompute(ws, aks_name)
except ComputeTargetException:
  # Create the cluster
  aks_target = ComputeTarget.create(workspace=ws, 
                                    name=aks_name, 
                                    provisioning_configuration=prov_config)

  aks_target.wait_for_completion(show_output = True)

print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# COMMAND ----------

# DBTITLE 1,Deploy AKS service
# Webservice creation using single command
from azureml.core.webservice import AksWebservice, Webservice
from azureml.exceptions import WebserviceException

try:
    webservice = AksWebservice(ws, 'los-aks')
    print(service.state)
except WebserviceException:
  # Set the web service configuration (using default here with app insights)
  aks_config = AksWebservice.deploy_configuration(enable_app_insights=True, compute_target_name=aks_name)

  # set the model path 
  model_path = "model"

  (webservice, model) = mlflow.azureml.deploy( model_uri='azureml://experiments/patient-los-sklearn/runs/c8d244ca-5166-4e77-bafd-71d365e920eb/artifacts/model',
                        workspace=ws,
                        model_name='LengthOfStaySklearnModel', 
                        service_name='los-aks', 
                        deployment_config=aks_config, 
                        tags=None, mlflow_home=None, synchronous=True)


  webservice.wait_for_deployment()

# COMMAND ----------

# DBTITLE 1,Score against the service
import requests
import json
from pyspark.sql import Row

test_df = X_test[:num_predictions]

# `sample_input` is a JSON-serialized pandas DataFrame with the `split` orientation
sample_input = {
    "columns": test_df.columns.tolist(),
    "data": test_df.values.tolist()
}

headers = {'Content-Type':'application/json'}

# Authenticate against the service.
if webservice.auth_enabled:
    headers['Authorization'] = 'Bearer '+ webservice.get_keys()[0]
    
response = requests.post(
              url=webservice.scoring_uri, data=json.dumps(sample_input),
              headers=headers)
response_json = json.loads(response.text)

rdd1 = sc.parallelize(response_json)
row_rdd = rdd1.map(lambda x: Row(x))
res=sqlContext.createDataFrame(row_rdd,['predictions'])
display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: looking ahead
# MAGIC 
# MAGIC With classical machine learning, we've been able to quickly build models to help Contoso Hospital better forecast patient bed availability. This is a reactive view though. How can we further help Contoso Hospital by *reducing* patient stay time?
# MAGIC 
# MAGIC One category worth investigation is our patients diagnosed with pneumonia. By supporting physicians with a model to accelerate diagnosis, we can accelerate treatment and reduce the length of stay. Let's start by looking at our data sets. 

# COMMAND ----------

# DBTITLE 1,Pneumonia patients & x-ray availability
from pyspark.sql import functions as F

prof_diag_smartd = spark.sql("SELECT * FROM lengthofstay_c")
xray_pneum = prof_diag_smartd.groupBy("pneum").agg(F.sum('xrayexamination'))
display(xray_pneum)

# COMMAND ----------

# DBTITLE 1,Check mounted x-ray directory
dbutils.fs.ls("dbfs:/mnt/xray/train/")

# COMMAND ----------

# DBTITLE 1,Display pneumonia x-ray images
image_df = spark.read.format("image").load("dbfs:/mnt/xray/train/PNEUMONIA/")

display(image_df) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Up
# MAGIC 
# MAGIC Uncomment and run the following cell to clean up after completing the notebook.

# COMMAND ----------

# DBTITLE 1,Archive and delete all models
#from mlflow.tracking.client import MlflowClient

#client = MlflowClient()
#for j in  ['LengthOfStaySklearnModel', 'LengthOfStaySparkModel']:
#  model_version_infos = client.search_model_versions("name = '%s'" % j)
#  for i in [model_version_info.version for model_version_info in model_version_infos]:
#    try:
#      client.transition_model_version_stage(
#        name=j,
#        version=i,
#        stage="Archived",
#      )
#      print('Archived...')
#    except:
#      print('Already Archived...')

  # Delete model
#  try:
#    client.delete_registered_model(name=j)
#    print('Deleted...')
#  except:
#    print('Already Deleted...')
