import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import kotlin.random.Random

fun main() {
    // Example data: house sizes, number of bedrooms, age of the house, distance from city center, and prices
    val houseData = arrayOf(
        floatArrayOf(1400f, 3f, 20f, 5f), floatArrayOf(1600f, 4f, 15f, 3f),
        floatArrayOf(1700f, 3f, 10f, 8f), floatArrayOf(1875f, 4f, 5f, 2f),
        floatArrayOf(1100f, 2f, 30f, 10f), floatArrayOf(1550f, 3f, 18f, 6f),
        floatArrayOf(2350f, 4f, 8f, 1f), floatArrayOf(2450f, 4f, 12f, 4f),
        floatArrayOf(1425f, 3f, 25f, 7f), floatArrayOf(1700f, 3f, 9f, 3f)
    )
    val housePrices = floatArrayOf(245000f, 312000f, 279000f, 308000f, 199000f, 219000f, 405000f, 324000f, 319000f, 299000f)

    // Preprocessing: Normalize features
    val preprocessor = PreprocessingStage {
        normalizer {
            meanAndStd()
        }
    }

    val dataset = OnHeapDataset.create(houseData, housePrices)
    val (train, test) = dataset.split(Random(123), 0.8)

    // Build the model
    val model = Sequential.of(
        Dense(128, inputShape = intArrayOf(4), activation = Activations.Relu),
        Dense(64, activation = Activations.Relu),
        Dense(32, activation = Activations.Relu),
        Dense(1, activation = Activations.Linear)
    )

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MSE,
            metric = Metrics.MAE
        )

        // Train the model
        it.fit(train, epochs = 200, batchSize = 5)

        // Evaluate the model
        val (testLoss, testMae) = it.evaluate(test)
        println("Test Loss: $testLoss, Test MAE: $testMae")

        // Predict with the model
        val prediction = it.predict(floatArrayOf(1500f, 3f, 20f, 5f))
        println("Predicted price for a house of 1500 sq ft, 3 bedrooms, 20 years old, 5 km from city center: $prediction")
    }
}
