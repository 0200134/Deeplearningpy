import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.ClassPredictionLabels;
import org.nd4j.linalg.api.ndarray.INDArray;

public class YOLOv5Detection {
    public static void main(String[] args) throws Exception {
        // Load YOLOv5 model
        ZooModel yoloModel = YOLO2.builder().build();
        ComputationGraph model = (ComputationGraph) yoloModel.initPretrained();

        // Load labels (Placeholder: replace with your actual label file)
        File labelsFile = new File("path/to/coco.names");
        String[] labels = ClassPredictionLabels.load(labelsFile);

        // Example input image path
        String imagePath = "path/to/image.jpg";

        // Perform detection
        performDetection(model, labels, imagePath);
    }
}import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ImagePreprocessing {
    public static INDArray preprocessImage(String imagePath) {
        // Load image using OpenCV
        Mat image = opencv_imgcodecs.imread(imagePath);

        // Preprocess the image (resize, normalize, etc.)
        // For YOLO, typically resize to 416x416 and normalize pixel values to [0, 1]
        Mat resizedImage = new Mat();
        opencv_imgproc.resize(image, resizedImage, new Size(416, 416));
        INDArray input = Nd4j.create(new int[] {1, 416, 416, 3});

        // Copy OpenCV Mat to ND4J INDArray
        for (int y = 0; y < 416; y++) {
            for (int x = 0; x < 416; x++) {
                double[] pixel = resizedImage.ptr(y, x).asDoublePointer().get();
                input.putScalar(new int[]{0, y, x, 0}, pixel[2] / 255.0);
                input.putScalar(new int[]{0, y, x, 1}, pixel[1] / 255.0);
                input.putScalar(new int[]{0, y, x, 2}, pixel[0] / 255.0);
            }
        }

        return input;
    }
}import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;

public class YOLOv5Detection {
    private static void performDetection(ComputationGraph model, String[] labels, String imagePath) throws Exception {
        // Preprocess the input image
        INDArray inputImage = ImagePreprocessing.preprocessImage(imagePath);

        // Perform inference
        INDArray[] output = model.output(inputImage);
        INDArray detections = output[0];

        // Process results
        Mat image = opencv_imgcodecs.imread(imagePath);
        processResults(detections, labels, image);

        // Display result
        CanvasFrame frame = new CanvasFrame("Object Detection");
        frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        frame.showImage(new Frame(image));
    }

    private static void processResults(INDArray detections, String[] labels, Mat image) {
        // Implement result processing and draw bounding boxes on the image
        // Parse detections and extract bounding boxes, classes, and confidence scores
        for (int i = 0; i < detections.size(1); i++) {
            float confidence = detections.getFloat(0, i, 4);
            if (confidence > 0.5) {
                int classId = Nd4j.argMax(detections.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all()).get(NDArrayIndex.interval(5, detections.size(2))), 1).getInt(0);
                float[] box = new float[] {
                    detections.getFloat(0, i, 0),
                    detections.getFloat(0, i, 1),
                    detections.getFloat(0, i, 2),
                    detections.getFloat(0, i, 3)
                };
                drawBoundingBox(image, box, labels[classId]);
            }
        }
    }

    private static void drawBoundingBox(Mat image, float[] box, String label) {
        int x = (int) (box[0] * image.cols());
        int y = (int) (box[1] * image.rows());
        int width = (int) (box[2] * image.cols());
        int height = (int) (box[3] * image.rows());

        opencv_imgproc.rectangle(image, new Point(x, y), new Point(x + width, y + height), new Scalar(0, 255, 0, 1));
        opencv_imgproc.putText(image, label, new Point(x, y - 10), opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0, 1), 2);
    }
}
