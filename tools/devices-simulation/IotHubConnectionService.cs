using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Azure.Devices;
using Microsoft.Azure.Devices.Client;
using Microsoft.Azure.Devices.Common.Exceptions;
using Newtonsoft.Json;

namespace AzureDatabricksDemo
{
  public class IotHubConnectionService
  {
    private readonly Random _random;
    private readonly string _connectionString;
    private readonly string _hostName;
    private readonly RegistryManager _registryManager;

    public IotHubConnectionService(string connectionString)
    {
      _random = new Random();
      _connectionString = connectionString;
      _hostName = connectionString.Split(";")[0].Split("=")[1];
      _registryManager = RegistryManager.CreateFromConnectionString(_connectionString);
    }

    public async Task<DeviceClient> GetOrCreateDeviceAsync(string deviceId)
    {
      var device = new Device(deviceId);
      try
      {
        device = await _registryManager.AddDeviceAsync(device);
      }
      catch (DeviceAlreadyExistsException)
      {
        device = await _registryManager.GetDeviceAsync(deviceId);
      }
      var deviceConnectionString = $"HostName={_hostName};DeviceId={deviceId};SharedAccessKey={device.Authentication.SymmetricKey.PrimaryKey}";
      return DeviceClient.CreateFromConnectionString(deviceConnectionString);
    }

    public async Task SendMessage(DeviceClient deviceClient, DeviceModel deviceModel, LengthOfStayRecord record)
    {
      try
      {
        var telemetryDataPoint = new Dictionary<string, Object>();
        telemetryDataPoint.Add(deviceModel.Properties[0].Name, record.Eid);
        telemetryDataPoint.Add(deviceModel.Properties[1].Name, record.Pulse);
        telemetryDataPoint.Add(deviceModel.Properties[2].Name, record.Respiration);
        telemetryDataPoint.Add(deviceModel.Properties[3].Name, record.BloodPressure);
        var messageString = JsonConvert.SerializeObject(telemetryDataPoint);
        Console.WriteLine($"Sending device update: {messageString}");
        var message = new Microsoft.Azure.Devices.Client.Message(Encoding.UTF8.GetBytes(messageString))
        {
          ContentType = "application/json",
          ContentEncoding = "utf-8"
        };
        await deviceClient.SendEventAsync(message);
      }
      catch (Exception e)
      {
        Console.WriteLine($"Error sending message to device {e}");
      }
    }
  }
}
