using System;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;
using System.IO;
using System.Windows.Forms.DataVisualization.Charting;

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
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"))
                .AppendCacheCheckpoint(_mlContext)
                .Append(_mlContext.Transforms.Concatenate("Features", "softmax2"))
                .Append(_mlContext.Regression.Trainers.Sdca());

            _model = pipeline.Fit(data);
            _mlContext.Model.Save(_model, data.Schema, _modelPath);
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
                ResultLabel.Text = $"Predicted Label: {prediction.PredictedLabel} - Score: {prediction.Score}";
                PictureBox.Image = image;
            }
        }

        private void EvaluateModel()
        {
            var testData = _mlContext.Data.LoadFromTextFile<ImageData>("imagesTest.tsv", separatorChar: '\t', hasHeader: true);
            var metrics = _mlContext.MulticlassClassification.Evaluate(_model.Transform(testData));

            LogMetrics(metrics);
        }

        private void LogMetrics(MulticlassClassificationMetrics metrics)
        {
            using (StreamWriter writer = new StreamWriter("metrics.log", true))
            {
                writer.WriteLine($"LogLoss: {metrics.LogLoss}");
                writer.WriteLine($"PerClassLogLoss: {string.Join(",", metrics.PerClassLogLoss)}");
            }
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
        private PictureBox PictureBox;

        private void InitializeComponent()
        {
            this.ResultLabel = new TextBox { Location = new System.Drawing.Point(12, 12), Size = new System.Drawing.Size(400, 22) };
            this.PredictButton = new Button { Location = new System.Drawing.Point(12, 40), Size = new System.Drawing.Size(100, 23), Text = "Predict Image" };
            this.PredictButton.Click += new System.EventHandler(this.PredictButton_Click);
            this.PictureBox = new PictureBox { Location = new System.Drawing.Point(12, 70), Size = new System.Drawing.Size(224, 224) };

            this.Controls.Add(this.ResultLabel);
            this.Controls.Add(this.PredictButton);
            this.Controls.Add(this.PictureBox);
            this.ClientSize = new System.Drawing.Size(460, 310);
            this.Name = "MainForm";
            this.Text = "Advanced AI App";
        }
    }
}
