namespace AzureDatabricksDemo
{
  public class SimulationOptions
  {
    public int SimulationTimeInSeconds { get; set; }

    public static SimulationOptions GetDefault()
    {
      return new SimulationOptions
      {
        SimulationTimeInSeconds = 60
      };
    }
  }
}
