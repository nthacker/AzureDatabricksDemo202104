-- Length of Stay

SELECT * FROM lengthofstay_c

-- Confirmed by Province_State

SELECT Province_State, SUM(Confirmed) FROM databricks_ds_daily_cases WHERE Country_Region = "US" GROUP BY Province_State ORDER BY SUM(Confirmed) DESC

-- Readmission Count

SELECT ReadmissionCount, COUNT(*) FROM lengthofstay_c GROUP BY ReadmissionCount; 

-- Average Length of Stay by Date

SELECT VisitDate, AVG(LengthOfStay)
FROM lengthofstay_c
GROUP BY VisitDate

-- Average Admission Rate

SELECT VisitDate, COUNT(PatientId)
FROM lengthofstay_c
GROUP BY VisitDate

-- Confirmed Cases

SELECT process_date, SUM(Confirmed)
FROM databricks_ds_daily_cases
WHERE Province_State = 'Washington'
GROUP BY process_date
ORDER BY process_date

-- OPTIMIZE Table

OPTIMIZE databricks_ds_daily_cases
ZORDER BY (process_date);

-- Trends

WITH DailyCases AS  
(  
    SELECT date_format(c.process_date, "MM-dd-yyyy") AS d1, SUM(c.Confirmed) / 8000 AS Confirmed
    FROM databricks_ds_daily_cases AS c
    WHERE c.Province_State = 'Washington' AND date_format(c.process_date, "MM-dd-yyyy") >= '10-01-2020' AND date_format(c.process_date, "MM-dd-yyyy") <= '12-01-2020'
    GROUP BY date_format(c.process_date, "MM-dd-yyyy")
  
), LengthOfStay AS  
(  
    SELECT date_format(VisitDate, "MM-dd-yyyy") AS d1, AVG(LengthOfStay) AS LengthOfStay
    FROM lengthofstay_c
    WHERE date_format(VisitDate, "MM-dd-yyyy") >= '10-01-2020' AND date_format(VisitDate, "MM-dd-yyyy") <= '12-01-2020'
    GROUP BY date_format(VisitDate, "MM-dd-yyyy")
)  
SELECT LengthOfStay.d1 AS Date, Confirmed, LengthOfStay
FROM DailyCases  
JOIN LengthOfStay
ON (DailyCases.d1 = LengthOfStay.d1)
ORDER BY LengthOfStay.d1