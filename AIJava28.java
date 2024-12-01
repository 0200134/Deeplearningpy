import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.TensorLabel;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;

public class ObjectDetectionActivity extends AppCompatActivity {

    private Interpreter tflite;
    private TensorLabel labeler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load the TensorFlow Lite model
        try {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(this, "model.tflite");
            tflite = new Interpreter(tfliteModel);

            // Load the labels
            labeler = new TensorLabel(FileUtil.loadLabelsFile(this, "labels.txt"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load an image and preprocess it
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.image);
        TensorImage inputImage = new TensorImage(TensorImage.DataType.FLOAT32);
        inputImage.load(bitmap);

        // Run inference
        Object[] output = new Object[1];
        tflite.run(inputImage.getBuffer(), output);
        float[][][] detections = (float[][][]) output[0];

        // Draw bounding boxes and labels on the image
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);

        for (float[] detection : detections) {
            float confidence = detection[2];
            if (confidence > 0.5) { // Adjust confidence threshold as needed
                int classId = (int) detection[1];
                String label = labeler.getMapWithClassNames().get(classId);
                RectF rect = new RectF(
                        detection[3] * bitmap.getWidth(),
                        detection[4] * bitmap.getHeight(),
                        detection[5] * bitmap.getWidth(),
                        detection[6] * bitmap.getHeight());
                canvas.drawRect(rect, paint);
                canvas.drawText(label, rect.left, rect.top, paint);
            }
        }

        // Display the image with bounding boxes
        ImageView imageView = findViewById(R.id.image_view);
        imageView.setImageBitmap(bitmap);
    }
}
// ... (similar to the previous code)

// ... (inside the onCreate method)

// Set up camera preview
CameraView cameraView = findViewById(R.id.camera_view);
cameraView.setListener(new CameraView.Listener() {
    @Override
    public void onFrame(Bitmap bitmap) {
        // Preprocess the image
        TensorImage inputImage = new TensorImage(TensorImage.DataType.FLOAT32);
        inputImage.load(bitmap);

        // Run inference
        Object[] output = new Object[1];
        tflite.run(inputImage.getBuffer(), output);
        float[][][] detections = (float[][][]) output[0];

        // Draw bounding boxes and labels on the image
        Canvas canvas = new Canvas(bitmap);
        // ... (similar to the previous code)

        // Display the image with bounding boxes
        cameraView.showFrame(bitmap);
    }
});
