using System;
using System.Collections.Generic;

namespace AzureDatabricksDemo
{
  public static class Constants
  {
    public const string TsiTimeZone = "Dateline Standard Time";
    public const string DateTimeFormat = "yyyy-MM-ddTHH:mm:ssZ";

    public const string SimulationStatusCreated = "Created";
    public const string SimulationStatusStarted = "Started";
    public const string SimulationStatusFinished = "Finished";
    public static readonly DeviceModel HubDeviceModel = new DeviceModel()
    {
      BaseName = string.Empty,
      Properties = new List<DeviceProperty>
      {
        new DeviceProperty() { Name = "Eid" },
        new DeviceProperty() { Name = "Pulse" },
        new DeviceProperty() { Name = "Respiration" },
        new DeviceProperty() { Name = "BloodPressure" }
      }
    };
  }
}
