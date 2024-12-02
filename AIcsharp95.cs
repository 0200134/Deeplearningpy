using System;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DeepLearningApp
{
    public partial class MainForm : Form
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;

        public MainForm()
        {
            InitializeComponent();
            _mlContext = new MLContext();
            TrainModel();
        }

        private void TrainModel()
        {
            var data = _mlContext.Data.LoadFromTextFile<ModelInput>("data.csv", separatorChar: ',', hasHeader: true);
            var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(ModelInput.Feature1), nameof(ModelInput.Feature2))
                        .Append(_mlContext.Regression.Trainers.Sdca());
            _model = pipeline.Fit(data);
        }

        private void PredictButton_Click(object sender, EventArgs e)
        {
            try
            {
                float feature1 = float.Parse(Feature1TextBox.Text);
                float feature2 = float.Parse(Feature2TextBox.Text);
                var input = new ModelInput { Feature1 = feature1, Feature2 = feature2 };
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
                float prediction = predictionEngine.Predict(input).Prediction;
                ResultLabel.Text = "Prediction: " + prediction;
            }
            catch (FormatException)
            {
                ResultLabel.Text = "Please enter valid numbers.";
            }
        }
    }

    public class ModelInput
    {
        [LoadColumn(0)]
        public float Feature1 { get; set; }
        [LoadColumn(1)]
        public float Feature2 { get; set; }
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
        private Label ResultLabel;
        private Button PredictButton;

        private void InitializeComponent()
        {
            this.Feature1TextBox = new TextBox { Location = new System.Drawing.Point(12, 12), Size = new System.Drawing.Size(100, 22) };
            this.Feature2TextBox = new TextBox { Location = new System.Drawing.Point(12, 40), Size = new System.Drawing.Size(100, 22) };
            this.ResultLabel = new Label { Location = new System.Drawing.Point(12, 69), Size = new System.Drawing.Size(52, 17), Text = "Result:" };
            this.PredictButton = new Button { Location = new System.Drawing.Point(12, 97), Size = new System.Drawing.Size(100, 23), Text = "Predict" };
            this.PredictButton.Click += new System.EventHandler(this.PredictButton_Click);

            this.Controls.Add(this.Feature1TextBox);
            this.Controls.Add(this.Feature2TextBox);
            this.Controls.Add(this.ResultLabel);
            this.Controls.Add(this.PredictButton);
            this.ClientSize = new System.Drawing.Size(284, 161);
            this.Name = "MainForm";
            this.Text = "Deep Learning App";
        }
    }
}
