# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Demo
# MAGIC ## Chapter # 3: Classical Machine Learning
# MAGIC ### MLflow Projects

# COMMAND ----------

# DBTITLE 1,Setup databricks-cli cfg file
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://adb-6739797150782991.11.azuredatabricks.net\ntoken = " + token, overwrite=True)

# COMMAND ----------

# DBTITLE 1,Define Backend configuration
backend = {
    "autoscale": {
        "min_workers": 2,
        "max_workers": 8
    },
    "spark_version": "7.4.x-cpu-ml-scala2.12",
    "node_type_id": "Standard_DS3_v2",
    "driver_node_type_id": "Standard_DS3_v2"
}

# COMMAND ----------

# DBTITLE 1,Run MLflow project from GitHub
import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "databricks"
mlflow.projects.run(uri="https://github.com/sebastian-zarmada/azure-databricks-mlflowproject-sample.git",
                    backend = "databricks",
                    backend_config = backend)
                    # experiment_id="3168170553969028") # Associate run with an existing experiment -> needs to be created from the UI