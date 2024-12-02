using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using System.Windows.Forms.DataVisualization.Charting;
using System.Net.Http;
using System.Net.Http.Json;

namespace AdvancedAIApp
{
    public partial class MainForm : Form
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private string _modelPath = "cnnModel.zip";

        public MainForm()
        {
            InitializeComponent();
            _mlContext = new MLContext();
            TrainModel();
        }

        private void TrainModel()
        {
            var data = _mlContext.Data.LoadFromTextFile<ImageData>("images.tsv", separatorChar: '\t', hasHeader: true);
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(_mlContext.Transforms.LoadImages("ImagePath", "ImagePath"))
                .Append(_mlContext.Transforms.ResizeImages("ImagePath", imageWidth: 224, imageHeight: 224))
                .Append(_mlContext.Transforms.ExtractPixels("ImagePath"))
                .Append(_mlContext.Model.LoadTensorFlowModel("model.pb")
                    .ScoreTensorName("softmax2")
                    .AddInput("input", "ImagePath")
                    .AddOutput("softmax2"))
                .Append(_mlContext.Transforms.Concatenate("Features", "softmax2"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

            var estimator = pipeline.Append(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
                                     .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainTestSplit = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            var model = estimator.Fit(trainingData);

            var transformedTrainingData = model.Transform(trainingData);
            var trainingMetrics = _mlContext.MulticlassClassification.Evaluate(transformedTrainingData);

            LogMetrics(trainingMetrics, "Training");
            var transformedTestData = model.Transform(testData);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(transformedTestData);

            LogMetrics(testMetrics, "Test");
            _mlContext.Model.Save(model, data.Schema, _modelPath);
            _model = model;
        }

        private void LoadModel()
        {
            DataViewSchema modelSchema;
            _model = _mlContext.Model.Load(_modelPath, out modelSchema);
        }

        private void PredictButton_Click(object sender, EventArgs e)
        {
            if (_model == null) LoadModel();

            var openFileDialog = new OpenFileDialog { Filter = "Image Files|*.jpg;*.jpeg;*.png" };
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var image = new Bitmap(openFileDialog.FileName);
                var input = new ImageData { ImagePath = openFileDialog.FileName };

                var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(_model);
                var prediction = predictionEngine.Predict(input);
                ResultLabel.Text = $"Predicted Label: {prediction.PredictedLabel} - Score: {string.Join(", ", prediction.Score)}";
                PictureBox.Image = image;
            }
        }

        private void LogMetrics(MulticlassClassificationMetrics metrics, string setType)
        {
            using (StreamWriter writer = new StreamWriter($"{setType}_metrics.log", true))
            {
                writer.WriteLine($"{setType} LogLoss: {metrics.LogLoss}");
                writer.WriteLine($"{setType} PerClassLogLoss: {string.Join(",", metrics.PerClassLogLoss)}");
            }
        }

        private async void DeployModel()
        {
            var client = new HttpClient();
            var modelContent = new ByteArrayContent(File.ReadAllBytes(_modelPath));
            var response = await client.PostAsync("http://yourserver.com/api/uploadmodel", modelContent);
            response.EnsureSuccessStatusCode();
        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }
        [LoadColumn(1)]
        public string Label { get; set; }
    }

    public class ImagePrediction : ImageData
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    partial class MainForm : Form
    {
        private TextBox ResultLabel;
        private Button PredictButton;
        private Button DeployButton;
        private PictureBox PictureBox;

        private void InitializeComponent()
        {
            this.ResultLabel = new TextBox { Location = new System.Drawing.Point(12, 12), Size = new System.Drawing.Size(400, 22) };
            this.PredictButton = new Button { Location = new System.Drawing.Point(12, 40), Size = new System.Drawing.Size(100, 23), Text = "Predict Image" };
            this.PredictButton.Click += new System.EventHandler(this.PredictButton_Click);
            this.DeployButton = new Button { Location = new System.Drawing.Point(12, 70), Size = new System.Drawing.Size(100, 23), Text = "Deploy Model" };
            this.DeployButton.Click += new System.EventHandler((sender, e) => DeployModel());
            this.PictureBox = new PictureBox { Location = new System.Drawing.Point(130, 12), Size = new System.Drawing.Size(224, 224) };

            this.Controls.Add(this.ResultLabel);
            this.Controls.Add(this.PredictButton);
            this.Controls.Add(this.DeployButton);
            this.Controls.Add(this.PictureBox);
            this.ClientSize = new System.Drawing.Size(460, 310);
            this.Name = "MainForm";
            this.Text = "Advanced AI App";
        }
    }
}
