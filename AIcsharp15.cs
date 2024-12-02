using System;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.AI.Luis;
using Microsoft.Bot.Schema;

namespace ChatbotExample
{
    public class SimpleBot : ActivityHandler
    {
        private readonly LuisRecognizer _luisRecognizer;

        public SimpleBot(LuisRecognizer luisRecognizer)
        {
            _luisRecognizer = luisRecognizer;
        }

        protected override async Task OnMessageActivityAsync(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken)
        {
            var recognizerResult = await _luisRecognizer.RecognizeAsync(turnContext, cancellationToken);
            var topIntent = recognizerResult.GetTopScoringIntent();

            switch (topIntent.intent)
            {
                case "Greeting":
                    await turnContext.SendActivityAsync(MessageFactory.Text("Hello! How can I help you today?"), cancellationToken);
                    break;
                case "Weather":
                    await turnContext.SendActivityAsync(MessageFactory.Text("Today's weather is sunny with a chance of rain later."), cancellationToken);
                    break;
                default:
                    await turnContext.SendActivityAsync(MessageFactory.Text("Sorry, I didn't understand that. Can you please rephrase?"), cancellationToken);
                    break;
            }
        }

        protected override async Task OnMembersAddedAsync(IList<ChannelAccount> membersAdded, ITurnContext<IConversationUpdateActivity> turnContext, CancellationToken cancellationToken)
        {
            foreach (var member in membersAdded)
            {
                if (member.Id != turnContext.Activity.Recipient.Id)
                {
                    await turnContext.SendActivityAsync(MessageFactory.Text("Welcome to the chatbot!"), cancellationToken);
                }
            }
        }
    }
}public class LuisSetup
{
    public static LuisRecognizer SetupLuis()
    {
        var luisApplication = new LuisApplication(
            "YourAppId",
            "YourSubscriptionKey",
            "https://YourRegion.api.cognitive.microsoft.com");

        var recognizerOptions = new LuisRecognizerOptionsV3(luisApplication)
        {
            PredictionOptions = new Microsoft.Azure.CognitiveServices.Language.LUIS.Runtime.Models.PredictionOptions
            {
                IncludeAllIntents = true,
                IncludeInstanceData = true
            }
        };

        return new LuisRecognizer(recognizerOptions);
    }
}
