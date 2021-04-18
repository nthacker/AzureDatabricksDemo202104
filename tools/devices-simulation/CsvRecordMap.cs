using CsvHelper.Configuration;

namespace AzureDatabricksDemo
{
  public sealed class CsvRecordMap : ClassMap<LengthOfStayRecord>
  {
    public CsvRecordMap() {  
      Map(x => x.Eid).Name("eid");
      Map(x => x.Pulse).Name("pulse");
      Map(x => x.Respiration).Name("respiration");
    }
  }
}
