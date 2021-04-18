using System;
using System.Collections.Generic;

namespace AzureDatabricksDemo
{
  public class Simulation
  {
    public string Status { get; set; }
    public DateTime StartedAt { get; set; }
    public DateTime FinishedAt { get; set; }
    public List<SimulatedDevice> Devices { get; set; }
  }

  public class SimulatedDevice
  {
    public string DeviceId { get; set; }
    public DeviceModel Model { get; set; }
    public int MilliSecondsBetweenGenerations { get; set; } = 100;
  }

  public class DeviceModel
  {
    public string BaseName { get; set; }
    public List<DeviceProperty> Properties { get; set;}
  }

  public class DeviceProperty
  {
    public string Name { get; set; }
  }

  public class LengthOfStayRecord
  {
    public int Eid { get; set; }
    public double Pulse { get; set;}
    public double Respiration { get; set;}
    public double BloodPressure { get; set; } = 0;
  }
}
