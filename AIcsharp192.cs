using System;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Windows.Forms.DataVisualization.Charting;

namespace DeepLearningApp
{
    public partial class MainForm : Form
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private string _modelPath = "model.zip";
        
        public MainForm()
        {
            InitializeComponent();
            _mlContext = new MLContext();
            LoadDataset();
        }

        private void LoadDataset()
        {
            var dataPath = "data.csv";
            if (File.Exists(dataPath))
            {
                var data = File.ReadAllLines(dataPath);
                foreach (var line in data)
                {
                    DataGridView.Rows.Add(line.Split(','));
                }
            }
        }

        private void TrainModel(string algorithm)
        {
            var data = _mlContext.Data.LoadFromTextFile<ModelInput>("data.csv", separatorChar: ',', hasHeader: true);
            var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(ModelInput.Feature1), nameof(ModelInput.Feature2), nameof(ModelInput.Feature3));

            if (algorithm == "SDCA")
            {
                pipeline = pipeline.Append(_mlContext.Regression.Trainers.Sdca());
            }
            else if (algorithm == "FastTree")
            {
                pipeline = pipeline.Append(_mlContext.Regression.Trainers.FastTree());
            }
            // Add more algorithms as needed

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
            try
            {
                float feature1 = float.Parse(Feature1TextBox.Text);
                float feature2 = float.Parse(Feature2TextBox.Text);
                float feature3 = float.Parse(Feature3TextBox.Text);
                var input = new ModelInput { Feature1 = feature1, Feature2 = feature2, Feature3 = feature3 };
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
                float prediction = predictionEngine.Predict(input).Prediction;
                ResultLabel.Text = "Prediction: " + prediction;
                LogPrediction(input, prediction);
                PlotPrediction(input, prediction);
            }
            catch (FormatException)
            {
                ResultLabel.Text = "Please enter valid numbers.";
            }
        }

        private void LogPrediction(ModelInput input, float prediction)
        {
            using (StreamWriter writer = new StreamWriter("predictions.log", true))
            {
                writer.WriteLine($"Input: {input.Feature1}, {input.Feature2}, {input.Feature3} - Prediction: {prediction}");
            }
        }

        private void PlotPrediction(ModelInput input, float prediction)
        {
            PredictionChart.Series["Predictions"].Points.AddXY(DateTime.Now.ToLongTimeString(), prediction);
        }
    }

    public class ModelInput
    {
        [LoadColumn(0)]
        public float Feature1 { get; set; }
        [LoadColumn(1)]
        public float Feature2 { get; set; }
        [LoadColumn(2)]
        public float Feature3 { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("Score")]
        public float Prediction { get; set; }
    }

    partial class MainForm : Form
    {
        private TextBox Feature1TextBox;
        private TextBox Feature2TextBox;
        private TextBox Feature3TextBox;
        private Label ResultLabel;
        private Button PredictButton;
        private Chart PredictionChart;
        private DataGridView DataGridView;
        private ComboBox AlgorithmComboBox;

        private void InitializeComponent()
        {
            this.Feature1TextBox = new TextBox { Location = new System.Drawing.Point(12, 12), Size = new System.Drawing.Size(100, 22) };
            this.Feature2TextBox = new TextBox { Location = new System.Drawing.Point(12, 40), Size = new System.Drawing.Size(100, 22) };
            this.Feature3TextBox = new TextBox { Location = new System.Drawing.Point(12, 68), Size = new System.Drawing.Size(100, 22) };
            this.ResultLabel = new Label { Location = new System.Drawing.Point(12, 97), Size = new System.Drawing.Size(52, 17), Text = "Result:" };
            this.PredictButton = new Button { Location = new System.Drawing.Point(12, 125), Size = new System.Drawing.Size(100, 23), Text = "Predict" };
            this.PredictButton.Click += new System.EventHandler(this.PredictButton_Click);
            this.PredictionChart = new Chart { Location = new System.Drawing.Point(150, 12), Size = new System.Drawing.Size(300, 300) };
            this.DataGridView = new DataGridView { Location = new System.Drawing.Point(12, 160), Size = new System.Drawing.Size(440, 150) };
            this.AlgorithmComboBox = new ComboBox { Location = new System.Drawing.Point(12, 320), Size = new System.Drawing.Size(100, 22) };
            this.AlgorithmComboBox.Items.AddRange(new object[] { "SDCA", "FastTree" });

            ChartArea chartArea = new ChartArea();
            this.PredictionChart.ChartAreas.Add(chartArea);
            Series series = new Series { Name = "Predictions", ChartType = SeriesChartType.Line };
            this.PredictionChart.Series.Add(series);

            this.Controls.Add(this.Feature1TextBox);
            this.Controls.Add(this.Feature2TextBox);
            this.Controls.Add(this.Feature3TextBox);
            this.Controls.Add(this.ResultLabel);
            this.Controls.Add(this.PredictButton);
            this.Controls.Add(this.PredictionChart);
            this.Controls.Add(this.DataGridView);
            this.Controls.Add(this.AlgorithmComboBox);
            this.ClientSize = new System.Drawing.Size(460, 360);
            this.Name = "MainForm";
            this.Text = "Deep Learning App";
        }
    }
}
