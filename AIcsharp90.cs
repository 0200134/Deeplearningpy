using System;
using System.Drawing;
using System.Windows.Forms;
using Tensorflow;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;

namespace MultiTaskLearningGUI
{
    public partial class MainForm : Form
    {
        private Model model;
        private TextBox textBox;
        private PictureBox pictureBox;
        private Button runButton;
        private Label textResultLabel;
        private Label imageResultLabel;

        public MainForm()
        {
            InitializeComponent();
            InitializeModel();
        }

        private void InitializeComponent()
        {
            this.Text = "Multi-Task Learning GUI";
            this.Size = new Size(600, 400);

            // Text input
            textBox = new TextBox() { Location = new Point(20, 20), Width = 500 };
            this.Controls.Add(textBox);

            // Image input
            pictureBox = new PictureBox() { Location = new Point(20, 60), Size = new Size(100, 100), BorderStyle = BorderStyle.Fixed3D };
            pictureBox.Click += new EventHandler(OnImageClick);
            this.Controls.Add(pictureBox);

            // Run button
            runButton = new Button() { Text = "Run Model", Location = new Point(20, 180) };
            runButton.Click += new EventHandler(OnRunButtonClick);
            this.Controls.Add(runButton);

            // Result labels
            textResultLabel = new Label() { Text = "Text Result:", Location = new Point(20, 220), Width = 500 };
            this.Controls.Add(textResultLabel);
            imageResultLabel = new Label() { Text = "Image Result:", Location = new Point(20, 260), Width = 500 };
            this.Controls.Add(imageResultLabel);
        }

        private void InitializeModel()
        {
            // Build and compile the multi-task learning model
            model = BuildMultiTaskModel();
            model.compile(optimizer: new Adam(0.0001), loss: new Dictionary<string, string>
            {
                { "text_output", "sparse_categorical_crossentropy" },
                { "image_output", "sparse_categorical_crossentropy" }
            }, metrics: new Dictionary<string, string[]>
            {
                { "text_output", new[] { "accuracy" } },
                { "image_output", new[] { "accuracy" } }
            });
        }

        private Model BuildMultiTaskModel()
        {
            var textInputIds = keras.Input(shape: (128,), dtype: tf.int32, name: "text_input_ids");
            var imageInputs = keras.Input(shape: (28, 28, 1), dtype: tf.float32, name: "image_inputs");

            // Text encoder using BERT
            var bertLayer = new TFBertModel("bert-base-uncased").call()[0];
            var textOutputs = bertLayer(textInputIds)[0];
            var textCLS = textOutputs[:, 0, :];
            var textDense = new Dense(64, activation: "relu").Apply(textCLS);
            var textOutput = new Dense(2, activation: "softmax", name: "text_output").Apply(textDense);

            // Image encoder using CNN
            var x = new Conv2D(32, (3, 3), activation: "relu").Apply(imageInputs);
            x = new MaxPooling2D((2, 2)).Apply(x);
            x = new Conv2D(64, (3, 3), activation: "relu").Apply(x);
            x = new MaxPooling2D((2, 2)).Apply(x);
            x = new Flatten().Apply(x);
            var imageDense = new Dense(64, activation: "relu").Apply(x);
            var imageOutput = new Dense(2, activation: "softmax", name: "image_output").Apply(imageDense);

            // Combined model
            var model = keras.Model(new Tensors(textInputIds, imageInputs), new Tensors(textOutput, imageOutput));
            model.summary();
            return model;
        }

        private void OnImageClick(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                pictureBox.Image = Image.FromFile(openFileDialog.FileName);
            }
        }

        private void OnRunButtonClick(object sender, EventArgs e)
        {
            // Preprocess inputs
            var textInput = textBox.Text;
            var (textInputIds, textAttentionMasks) = TokenizeAndPreprocess(new[] { textInput });

            var imageInput = PreprocessImage(pictureBox.Image);

            // Predict
            var outputs = model.predict(new Dictionary<string, NDArray>
            {
                { "text_input_ids", textInputIds },
                { "image_inputs", imageInput }
            });

            var textResult = outputs["text_output"].argmax();
            var imageResult = outputs["image_output"].argmax();

            // Display results
            textResultLabel.Text = $"Text Result: {textResult}";
            imageResultLabel.Text = $"Image Result: {imageResult}";
        }

        static (NDArray, NDArray) TokenizeAndPreprocess(string[] texts)
        {
            var tokenizer = new BertTokenizer("bert-base-uncased");

            List<long[]> inputIdsList = new List<long[]>();
            List<long[]> attentionMasksList = new List<long[]>();

            foreach (var text in texts)
            {
                var tokens = tokenizer.Encode(text);
                inputIdsList.Add(tokens.InputIds);
                attentionMasksList.Add(tokens.AttentionMask);
            }

            var inputIds = np.array(inputIdsList);
            var attentionMasks = np.array(attentionMasksList);
            return (inputIds, attentionMasks);
        }

        static NDArray PreprocessImage(Image image)
        {
            Bitmap bitmap = new Bitmap(image, new Size(28, 28));
            var data = new float[28, 28, 1];
            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    data[x, y, 0] = bitmap.GetPixel(x, y).R / 255.0f;
                }
            }
            return np.array(new float[,,,] { { data } });
        }
    }
}
