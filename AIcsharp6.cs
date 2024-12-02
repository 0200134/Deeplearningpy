using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace CognitiveServicesExample
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var subscriptionKey = "YOUR_SUBSCRIPTION_KEY";
            var endpoint = "https://YOUR_REGION.api.cognitive.microsoft.com/text/analytics/v3.0/sentiment";

            var client = new HttpClient();
            client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", subscriptionKey);

            var documents = new
            {
                documents = new[]
                {
                    new { language = "en", id = "1", text = "Hello world. This is a test of the cognitive services text analytics API." }
                }
            };

            var json = Newtonsoft.Json.JsonConvert.SerializeObject(documents);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync(endpoint, content);
            var result = await response.Content.ReadAsStringAsync();

            Console.WriteLine(result);
        }
    }
}
