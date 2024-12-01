import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.TensorLabel;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class MainActivity extends AppCompatActivity {

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
        float[][] output = new float[1][10]; // Adjust output shape as needed
        tflite.run(inputImage.getBuffer(), output);

        // Get the top-predicted label
        String label = labeler.getMapWithClassNames().get(argmax(output[0]));

        // Display the result
        ImageView imageView = findViewById(R.id.image_view);
        imageView.setImageBitmap(bitmap);

        TextView textView = findViewById(R.id.result_text);
        textView.setText("Predicted class: " + label);
    }

    // Helper function to find the index of the highest value in an array
    private int argmax(float[] array) {
        int bestIndex = 0;
        float bestValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > bestValue) {
                bestIndex = i;
                bestValue = array[i];
            }
        }
        return bestIndex;
    }
}
