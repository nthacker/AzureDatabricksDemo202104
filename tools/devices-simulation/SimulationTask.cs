using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Azure.Devices.Client;

namespace AzureDatabricksDemo
{
  public class SimulationTask
  {
    private readonly Simulation _simulation;
    private readonly SimulationOptions _options;
    private readonly List<LengthOfStayRecord> _records;
    private readonly Random _random;
    private readonly int _minutesToRun;
    private IotHubConnectionService _iotHubConnectionService;
    private Dictionary<string, DeviceClient> _deviceClients;

    public SimulationTask(SimulationOptions options, List<LengthOfStayRecord> records, IotHubConnectionService iotHubConnectionService, int minutesToRun)
    {
      _options = options;
      _iotHubConnectionService = iotHubConnectionService;
      _records = records;
      _minutesToRun = minutesToRun;
      _random = new Random();
      _deviceClients = new Dictionary<string, DeviceClient>();

      // Using single device for all records
      var devices = new List<SimulatedDevice>();
      devices.Add(new SimulatedDevice()
      {
        DeviceId = "Main",
        Model = Constants.HubDeviceModel
      });
      _simulation = new Simulation()
      {
        Devices = devices,
        Status = Constants.SimulationStatusCreated
      };
    }

    public async Task Start()
    {
      // Set initial generation time
      Console.WriteLine("Initializing devices...");
      DeviceClient deviceClient = null;
      foreach (var device in _simulation.Devices)
      {
        deviceClient = await _iotHubConnectionService.GetOrCreateDeviceAsync(device.DeviceId);
        _deviceClients.Add(device.DeviceId, deviceClient);
        Console.WriteLine($"Device {device.DeviceId} initialized.");
      }
      Console.WriteLine("Devices initialized.");
      _simulation.StartedAt = DateTime.Now;
      _simulation.Status = Constants.SimulationStatusStarted;
      var model = _simulation.Devices[0].Model;
      foreach (var patientRecordsItem in _records)
      {
#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        _iotHubConnectionService.SendMessage(deviceClient, model, patientRecordsItem);
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        await Task.Delay(_simulation.Devices[0].MilliSecondsBetweenGenerations);
      }
      _simulation.Status = Constants.SimulationStatusFinished;
      _simulation.FinishedAt = _simulation.StartedAt.AddSeconds(_minutesToRun * 60);
      Console.WriteLine("Simulation Finished, press the `Escape` key to continue.");
    }

    public void Stop()
    {
      _simulation.FinishedAt = DateTime.Now;
    }

    public Simulation GetSimulation()
    {
      return _simulation;
    }

    public void ResetEndTime()
    {
       _simulation.FinishedAt = DateTime.Now.AddSeconds(_minutesToRun * 60);
    }
  }
}
