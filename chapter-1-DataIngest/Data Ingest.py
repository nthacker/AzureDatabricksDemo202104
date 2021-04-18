# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks Healthcare Demo
# MAGIC ## Chapter 1: Data

# COMMAND ----------

# MAGIC %md
# MAGIC ![image info](https://visualassets.blob.core.windows.net/flow-diagrams/data-ingest.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC Around the world, healthcare providers have been pushed to their limits and beyond by the COVID-19 pandemic. Scarce resources, like PPE and hospital beds, need to be carefully managed to deliver the best possible outcomes for patients.
# MAGIC 
# MAGIC Let's start by seeing how Contoso Hospital can use Azure Databricks to quickly bring together data sources to build a modern, high-performance data lake in order to optimize their resources and maximize patient services.
# MAGIC 
# MAGIC In a pandemic, being able to forecast bed availability is critical. Let's start there.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patient data
# MAGIC 
# MAGIC Our patient profile data is in Cosmos DB. Let's fetch it using the native support in Azure Databricks so we can see things like admissions, readmissions, and length of stay.
# MAGIC 
# MAGIC Length of stay will be our focus but we'll need to bring in data from multiple systems to truly understand the contributing factors.

# COMMAND ----------

# DBTITLE 1,Connect to Cosmos DB
connectionConfig = {
  "Endpoint" : "https://databricks-demo-cosmosdb.documents.azure.com:443/",
  "Masterkey" : "nXBTOWhV7GzdetjMqwU990XMo01NGXrPlpHgIGdbPX3JkFZDmOMyWjgPOfIE3psPBScV30qMuWtlhPpn6fSUig==",
  "Database" : "Patients",
  "preferredRegions" : "West US 2",
  "Collection": "profiles",
  "schema_samplesize" : "1000",
  "query_pagesize" : "200000",
  "query_custom" : "SELECT * FROM c"
}

# COMMAND ----------

# DBTITLE 1,Load patient profile data
patient_profiles = spark.read.format("com.microsoft.azure.cosmosdb.spark").options(**connectionConfig).load()
patient_profiles.alias("patient_profiles")
display(patient_profiles)

# COMMAND ----------

# DBTITLE 1,Save patient profile data to Delta Lake
patient_profiles.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/patient_profiles")
spark.sql("DROP TABLE IF EXISTS patient_profiles")
spark.sql("CREATE TABLE patient_profiles USING DELTA LOCATION '/mnt/delta/patient_profiles/'")
patient_profiles = spark.sql("SELECT * FROM patient_profiles")
display(patient_profiles)

# COMMAND ----------

# DBTITLE 1,Question: how many admissions do get per day?
from pyspark.sql import functions as F
display(patient_profiles.groupBy("VisitDate").agg(F.count('PatientId')))

# COMMAND ----------

# DBTITLE 1,Question: how many readmissions are normal?
display(patient_profiles.groupBy("ReadmissionCount").agg(F.count('PatientId')).orderBy("ReadmissionCount", ascending=True))

# COMMAND ----------

# DBTITLE 1,Question: are we seeing patients stay longer?
display(patient_profiles.groupBy("VisitDate").agg(F.mean("LengthOfStay")).orderBy("VisitDate", ascending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnostics data
# MAGIC 
# MAGIC From our initial dataset, we're seeing patient length of stay increase. Let's see if we can explain it by fetching information about patient diagnostics.
# MAGIC 
# MAGIC Native support for Azure Data Lake makes it easy to pull in semi-structured data files like diagnostic CSV exports.

# COMMAND ----------

# DBTITLE 1,Mount Azure Data Lake Storage
dbutils.fs.unmount("/mnt/datalake")

configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": "d2e55094-6fc0-439c-bbc9-b56e93d194ef",
           "fs.azure.account.oauth2.client.secret": "1O.iRFbweB4fAoP2IAouO0.~3q..Wq3v7w",
           "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/token"}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://diagnostics@databricksdatalakestg.dfs.core.windows.net/",
  mount_point = "/mnt/datalake",
  extra_configs = configs)

dbutils.fs.ls("/mnt/datalake")

# COMMAND ----------

# DBTITLE 1,Fetch diagnostics data
patient_diagnostics = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/datalake/datalake_diagnostics.csv")
display(patient_diagnostics)

# COMMAND ----------

# DBTITLE 1,Write diagnostics data to Delta Lake
patient_diagnostics.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/patient_diagnostics")
spark.sql("DROP TABLE IF EXISTS patient_diagnostics")
spark.sql("CREATE TABLE patient_diagnostics USING DELTA LOCATION '/mnt/delta/patient_diagnostics/'")
patient_diagnostics = spark.sql("SELECT * FROM patient_diagnostics")
display(patient_diagnostics)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have diagnostics data, let's use Databricks to join our datasets on the patient identifier. 

# COMMAND ----------

# DBTITLE 1,Join patient with diagnostics data
profiles_diagnostics = patient_profiles.join(patient_diagnostics, F.col("PatientId") == F.col("default.patient_diagnostics.eid"), "inner").select('Alcohol','DischargedDate','FirstName','Gender','Height','LastName','LengthOfStay','PatientId','ReadmissionCount','Smoker','VisitDate','Weight','eid', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo','xrayexamination')
display(profiles_diagnostics)

# COMMAND ----------

# DBTITLE 1,Question: how does the patient length of stay change by diagnosis?
pneum_los = profiles_diagnostics.groupBy("pneum","asthma","irondef").agg(F.mean('LengthOfStay'))
display(pneum_los)

# COMMAND ----------

# MAGIC %md
# MAGIC ## IoT Data
# MAGIC 
# MAGIC Modern healthcare facilities, and modern businesses more general, are seeing an increased adoption of IoT devices providing a real-time perspective on operations. Contoso uses IoT-based solutions to monitor patient pulse and respiration so let's pull this data in to so we can improve our picture of each patient.
# MAGIC 
# MAGIC IoT data is different to what we've seen so far. It's a streaming, real-time data source but fortunately we can use Delta Lake to store and query it just as easily as our existing data sets.

# COMMAND ----------

# DBTITLE 1,Connect to IoT Hub
from pyspark.sql.types import *

# Read Event Hub's stream
conf = {}
conf["eventhubs.connectionString"] = "Endpoint=sb://iothub-ns-databricks-6648512-6938c21b2f.servicebus.windows.net/;SharedAccessKeyName=iothubowner;SharedAccessKey=4uqmdfa7JG/Y1jXe77AXjWyYPir3LlnxC5TZoSkQWt4=;EntityPath=databricksiothub"
read_df = spark.readStream.format("eventhubs").options(**conf).load()
read_schema = StructType([
  StructField("Eid", StringType(), True),
  StructField("Pulse", StringType(), True),
  StructField("Respiration", StringType(), True)])
decoded_df = read_df.select(F.from_json(F.col("body").cast("string"), read_schema).alias("Readings"))
decoded_df = decoded_df.select(F.col("Readings").getItem("Pulse").alias("Pulse"), F.col("Readings").getItem("Respiration").alias("Respiration"), F.col("Readings").getItem("Eid").alias("Eid"))
display(decoded_df)

# COMMAND ----------

# DBTITLE 1,Stream IoT data into Delta Lake
decoded_df.writeStream.format("delta").outputMode("append").option("checkpointLocation", "/delta/events/_checkpoints/etl-from-json").option("overwriteSchema", "true").option("mergeSchema", "true").start("/mnt/delta/smartdevices")

# COMMAND ----------

# DBTITLE 1,Mount IoT data in Delta Lake
spark.sql("DROP TABLE IF EXISTS smartdevices")
spark.sql("CREATE TABLE smartdevices USING DELTA LOCATION '/mnt/delta/smartdevices/'")
smartdevices_deltalake = spark.sql("SELECT Eid, Pulse, Respiration FROM smartdevices")
smartdevices_deltalake.alias("smartdevices_deltalake")
display(smartdevices_deltalake)

# COMMAND ----------

# DBTITLE 1,Question: what's the average respiration & average pulse of our patients?
display(smartdevices_deltalake.agg(F.mean('Pulse'), F.mean('Respiration')))

# COMMAND ----------

# DBTITLE 1,Question: how do pulse and respiration vary across common diagnostics for readmitted patients?
prof_diag_smartd = profiles_diagnostics.join(smartdevices_deltalake, F.col("PatientId") == F.col("default.smartdevices.Eid"), "inner").select('Alcohol','DischargedDate','FirstName','Gender','Height','LastName','LengthOfStay','PatientId','ReadmissionCount','Smoker','VisitDate','Weight','dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo','xrayexamination','Pulse','Respiration')
avg_pulse = prof_diag_smartd.filter(prof_diag_smartd.ReadmissionCount == "4").groupBy("pneum", "asthma", "irondef").agg(F.mean('Pulse'),F.mean('Respiration'))
display(avg_pulse)

# COMMAND ----------

# MAGIC %md
# MAGIC We now have a great picture of our patients and their overall health. Let's add this dataset to Delta Lake for further analysis.

# COMMAND ----------

# DBTITLE 1,Write consolidated data to Delta Lake
prof_diag_smartd.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/lengthofstay_c")
spark.sql("DROP TABLE IF EXISTS lengthofstay_c")
spark.sql("CREATE TABLE lengthofstay_c USING DELTA LOCATION '/mnt/delta/lengthofstay_c/'")
consolidated_data = spark.sql("SELECT * FROM lengthofstay_c")
display(consolidated_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## External data
# MAGIC 
# MAGIC Our internal data sources don't sufficiently explain the changes in admission rates. Let's pull in a publicly available data source from the Databricks datasets collection, the COVID-19 dataset, to better understand operational trends.
# MAGIC 
# MAGIC The dataset has a variety of schemas so we can use Databricks and its powerful data wrangling capabilities to bring it into alignment.

# COMMAND ----------

# DBTITLE 1,Fetch COVID-19 data from Databricks
import os
import pandas as pd
import glob
from pyspark.sql.functions import input_file_name, lit, col

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, TimestampType
schema = StructType([
  StructField('FIPS', IntegerType(), True), 
  StructField('Admin2', StringType(), True),
  StructField('Province_State', StringType(), True),  
  StructField('Country_Region', StringType(), True),  
  StructField('Last_Update', TimestampType(), True),  
  StructField('Lat', DoubleType(), True),  
  StructField('Long_', DoubleType(), True),
  StructField('Confirmed', IntegerType(), True), 
  StructField('Deaths', IntegerType(), True), 
  StructField('Recovered', IntegerType(), True), 
  StructField('Active', IntegerType(), True),   
  StructField('Combined_Key', StringType(), True),  
  StructField('process_date', DateType(), True),    
])

# Create initial empty Spark DataFrame based on preceding schema
daily_cases = spark.createDataFrame([], schema)

# Creates a list of all csv files
globbed_files = glob.glob("/dbfs/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/*.csv") 

i = 0
for csv in globbed_files:
  # Filename
  source_file = csv[5:200]
  process_date = csv[100:104] + "-" + csv[94:96] + "-" + csv[97:99]
  
  # Read data into temporary dataframe
  df_tmp = spark.read.option("inferSchema", True).option("header", True).csv(source_file)
  df_tmp.createOrReplaceTempView("df_tmp")

  # Obtain schema
  schema_txt = ' '.join(map(str, df_tmp.columns)) 
  
  # Three schema types (as of 2020-04-08) 
  schema_01 = "Province/State Country/Region Last Update Confirmed Deaths Recovered" # 01-22-2020 to 02-29-2020
  schema_02 = "Province/State Country/Region Last Update Confirmed Deaths Recovered Latitude Longitude" # 03-01-2020 to 03-21-2020
  schema_03 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key" # 03-22-2020 to
  schema_04 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key Incidence_Rate Case-Fatality_Ratio"
  schema_05 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key Incident_Rate Case_Fatality_Ratio"
  
  # Insert data based on schema type
  if (schema_txt == schema_01):
    df_tmp = (df_tmp
                .withColumn("FIPS", lit(None).cast(IntegerType()))
                .withColumn("Admin2", lit(None).cast(StringType()))
                .withColumn("Province_State", col("Province/State"))
                .withColumn("Country_Region", col("Country/Region"))
                .withColumn("Last_Update", col("Last Update"))
                .withColumn("Lat", lit(None).cast(DoubleType()))
                .withColumn("Long_", lit(None).cast(DoubleType()))
                .withColumn("Active", lit(None).cast(IntegerType()))
                .withColumn("Combined_Key", lit(None).cast(StringType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        "process_date")
               )
    daily_cases = daily_cases.union(df_tmp)
    
  elif (schema_txt == schema_02):
    df_tmp = (df_tmp
                .withColumn("FIPS", lit(None).cast(IntegerType()))
                .withColumn("Admin2", lit(None).cast(StringType()))
                .withColumn("Province_State", col("Province/State"))
                .withColumn("Country_Region", col("Country/Region"))
                .withColumn("Last_Update", col("Last Update"))
                .withColumn("Lat", col("Latitude"))
                .withColumn("Long_", col("Longitude"))
                .withColumn("Active", lit(None).cast(IntegerType()))
                .withColumn("Combined_Key", lit(None).cast(StringType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        "process_date")
               )
    daily_cases = daily_cases.union(df_tmp)
    
  elif (schema_txt == schema_04 or schema_txt == schema_05):
    df_tmp = (df_tmp
                .withColumn("FIPS", lit(None).cast(IntegerType()))
                .withColumn("Admin2", lit(None).cast(StringType()))
                .withColumn("Province_State", col("Province_State"))
                .withColumn("Country_Region", col("Country_Region"))
                .withColumn("Last_Update", col("Last_Update"))
                .withColumn("Lat", col("Lat"))
                .withColumn("Long_", col("Long_"))
                .withColumn("Active", lit(None).cast(IntegerType()))
                .withColumn("Combined_Key", lit(None).cast(StringType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        "process_date")
               )
    daily_cases = daily_cases.union(df_tmp)

  elif (schema_txt == schema_03):
    df_tmp = df_tmp.withColumn("process_date", lit(process_date))
    daily_cases = daily_cases.union(df_tmp)
  else:
    print("Schema may have changed")
    print(df_tmp.columns)
    raise
  
  # print out the schema being processed by date
  print("%s | %s" % (process_date, schema_txt))

# COMMAND ----------

# DBTITLE 1,Write dataset to Delta Lake
daily_cases.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/databricks_ds_daily_cases")
spark.sql("DROP TABLE IF EXISTS databricks_ds_daily_cases")
spark.sql("CREATE TABLE databricks_ds_daily_cases USING DELTA LOCATION '/mnt/delta/databricks_ds_daily_cases/'")
new_data = spark.sql("SELECT * FROM databricks_ds_daily_cases")
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Working with data
# MAGIC 
# MAGIC Now that we have our data, let's start analyze it. Databricks make it easy to ask complex questions over big datasets. Let's see an example of this with the Spark API.

# COMMAND ----------

# DBTITLE 1,Question: what are the confirmed cases in Washington? (Spark edition)
display(daily_cases.filter(daily_cases.Province_State == 'Washington').groupBy("process_date").agg(F.sum("Confirmed")).orderBy("process_date", ascending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC While data can be manipulated directly via the Spark dataframe, many data scientists prefer to use pandas. Let's see the same query in pandas.

# COMMAND ----------

# DBTITLE 1,Question: what are the confirmed cases in Washington? (pandas edition)
import pandas as pd

daily_cases_pandas = daily_cases.toPandas()
daily_cases_pandas = daily_cases_pandas[daily_cases_pandas['Province_State'] == 'Washington'].groupby(['process_date'])['Confirmed'].sum().to_frame().reset_index().sort_values(by=['process_date'])
display(daily_cases_pandas.plot.line(x = 'process_date', y = 'Confirmed'))

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately, as you might have spotted, using pandas means we give up the scale out, big data capabilities of Spark as we need to load it onto a single node. To resolve this issue, Databricks provides Koalas, which implements the pandas API over Spark.
# MAGIC 
# MAGIC Here you can see the same query with identical code but leveraging Spark's big data capabilities under the covers.

# COMMAND ----------

# DBTITLE 1,Question: what are the confirmed cases in Washington? (koalas edition)
import databricks.koalas as ks

daily_cases_koalas = ks.read_delta('/mnt/delta/databricks_ds_daily_cases')
daily_cases_koalas = daily_cases_koalas[daily_cases_koalas['Province_State'] == 'Washington'].groupby(['process_date'])['Confirmed'].sum().to_frame().reset_index().sort_values(by=['process_date'])
display(daily_cases_koalas.plot.line(x = 'process_date', y = 'Confirmed'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next steps: analytics
# MAGIC 
# MAGIC Now that we've brought together the data that we'll need to understand patient stay lengths - from operational data like patient profiles and diagnostics, through to real-time streaming data and external pandemic information - the next step will be to analyze it using SQL Analytics.
# MAGIC 
# MAGIC If we can identify the drivers and trends that influence how long patients stay in our hospital, we'll be able to better forecast our capacity in the coming months.
