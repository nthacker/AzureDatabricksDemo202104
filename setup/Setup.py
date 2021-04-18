# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Demo
# MAGIC ## Setup Notebook

# COMMAND ----------

# DBTITLE 1,Setup
import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core
from azureml.core.authentication import InteractiveLoginAuthentication

from azureml.core import Workspace

try:    
    interactive_auth = InteractiveLoginAuthentication(tenant_id="<tenant_id>")
    # Get instance of the Workspace and write it to config file
    ws = Workspace(
        subscription_id = '<subscription_id>', 
        resource_group = '<resource_group_name>', 
        workspace_name = '<workspace_name>',
        auth = interactive_auth)
    # Writes workspace config file
    ws.write_config()
    
    print('Library configuration succeeded')
except Exception as e:
    print(e)
    print('Workspace not found')

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://training@<storage_account_name>.blob.core.windows.net/chest_xray",
  mount_point = "/mnt/xray",
  extra_configs = {"fs.azure.account.key.lengthofstaystorage.blob.core.windows.net":"<access_key>"})

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.<storage_account_name>.blob.core.windows.net",
  "<access_key>")

# COMMAND ----------

los_sparkDF = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("wasbs://training@<storage_account_name>.blob.core.windows.net/LengthOfStay.csv")
los_sparkDF.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/lengthofstay")
spark.sql("DROP TABLE IF EXISTS lengthofstay")
spark.sql("CREATE TABLE lengthofstay USING DELTA LOCATION '/mnt/delta/lengthofstay/'")