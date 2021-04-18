# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Healthcare Demo
# MAGIC ## Chapter 5: Business Analytics

# COMMAND ----------

# MAGIC %md
# MAGIC ![image info](https://visualassets.blob.core.windows.net/flow-diagrams/business-analytics.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC By bringing together their existing data assets into Delta Lake, Contoso Hospital has been able to analyze and respond to the pandemic with improved forecasting clarity and optimized diagnostics. For many of their users though, these massive raw datasets describing individual patients are of little value and need to be presented in a more consumable, aggregated format for business planning.
# MAGIC 
# MAGIC Let's see how we can package this data and insert it into our corporate data warehouse for our business users to access.

# COMMAND ----------

# DBTITLE 1,Read batch predictions from Delta Lake
predicted_data = spark.sql("SELECT * FROM predictions")
display(predicted_data)

# COMMAND ----------

import databricks.koalas as ks
import numpy as np

koalas_df = ks.DataFrame(predicted_data)
display(koalas_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze

# COMMAND ----------

# DBTITLE 1,Question: what are the key statistics of our patient dataset?
koalas_df.describe()

# COMMAND ----------

# DBTITLE 1,Question: what's the percentage occupation by day?
def fillXrayExamination(row):
    return (row['prediction'] / 420) * 100

ks.set_option('compute.ops_on_diff_frames', True)
occupation_by_day = koalas_df[['vdate', 'prediction']].groupby(['vdate'])['prediction'].sum().to_frame().reset_index()
occupation_by_day['vdate'] = ks.to_datetime(occupation_by_day['vdate'])
occupation_by_day['percent_occupation'] = occupation_by_day.apply(fillXrayExamination, axis=1)
occupation_by_day.head(10)

# COMMAND ----------

display(occupation_by_day.sort_values(by='vdate', ascending=False).head(50).to_spark())

# COMMAND ----------

# DBTITLE 1,Question: what's the average length of stay by month?
avg_los_per_month = koalas_df[['vdate', 'prediction']]
avg_los_per_month['vdate'] = ks.to_datetime(avg_los_per_month['vdate'])
avg_los_per_month['month'] = avg_los_per_month.apply(lambda x: x['vdate'].month, axis=1)
avg_los_per_month = avg_los_per_month.groupby(['month'])['prediction'].mean().to_frame().reset_index()
avg_los_per_month.head(5)

# COMMAND ----------

display(avg_los_per_month.sort_values(by='month', ascending=True).to_spark())

# COMMAND ----------

# DBTITLE 1,Question: what's the average length of stay by facility?
avg_los_gm = koalas_df[['vdate', 'facid', 'prediction']]
avg_los_gm['vdate'] = ks.to_datetime(avg_los_gm['vdate'])
avg_los_gm['month'] = avg_los_gm.apply(lambda x: x['vdate'].month, axis=1)
avg_los_gm = avg_los_gm.groupby(['month','facid'])['prediction'].mean().to_frame().reset_index()
avg_los_gm.head(5)

# COMMAND ----------

display(avg_los_gm.sort_values(by='month', ascending=True).to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move to Data Warehouse
# MAGIC 
# MAGIC With native support for Azure Synapse Analytics (former SQL Data Warehouse), it's easy for us to move our analysis into the data warehouse for a broader set of users to access.

# COMMAND ----------

# DBTITLE 1,Set up the blob storage account access key
spark.conf.set(
  "fs.azure.account.key.lengthofstaystorage.blob.core.windows.net",
  "HXz5DC1PSOgQLGr9tB6KoDZCdrXxgxc40MuRtdYYzdm4k10R5eGE5w5++uN8SrUPAxVZbpSORvKLeiQR/qJhRA==")

# COMMAND ----------

# DBTITLE 1,Write datasets to data warehouse
occupation_by_day.to_spark().write \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://synapse-azure-databricks-demo.sql.azuresynapse.net:1433;database=SynapseDW;user=sqladminuser@synapse-azure-databricks-demo;password=Password123!;loginTimeout=30;") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", "DailyOcupation") \
  .option("tempDir", "wasbs://analytics@lengthofstaystorage.blob.core.windows.net/temp") \
  .save()

# COMMAND ----------

avg_los_per_month.to_spark().write \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://synapse-azure-databricks-demo.sql.azuresynapse.net:1433;database=SynapseDW;user=sqladminuser@synapse-azure-databricks-demo;password=Password123!;loginTimeout=30;") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", "MonthlyLengthOfStay") \
  .option("tempDir", "wasbs://analytics@lengthofstaystorage.blob.core.windows.net/temp") \
  .save()

# COMMAND ----------

avg_los_gm.to_spark().write \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://synapse-azure-databricks-demo.sql.azuresynapse.net:1433;database=SynapseDW;user=sqladminuser@synapse-azure-databricks-demo;password=Password123!;loginTimeout=30;") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", "FacilityLengthOfStay") \
  .option("tempDir", "wasbs://analytics@lengthofstaystorage.blob.core.windows.net/temp") \
  .save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC Our key datasets are now in our corporate data warehouse. Contoso Hospital analysts can now combine these insights with other operational datasets to plan the operation of the hospital.
