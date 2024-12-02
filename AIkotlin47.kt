import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

fun main() {
    // Example data: house sizes, number of bedrooms, and prices
    val houseData = arrayOf(
        floatArrayOf(1400f, 3f), floatArrayOf(1600f, 4f), floatArrayOf(1700f, 3f), floatArrayOf(1875f, 4f),
        floatArrayOf(1100f, 2f), floatArrayOf(1550f, 3f), floatArrayOf(2350f, 4f), floatArrayOf(2450f, 4f),
        floatArrayOf(1425f, 3f), floatArrayOf(1700f, 3f)
    )
    val housePrices = floatArrayOf(245000f, 312000f, 279000f, 308000f, 199000f, 219000f, 405000f, 324000f, 319000f, 299000f)

    val dataset = OnHeapDataset.create(houseData, housePrices)

    val model = Sequential.of(
        Dense(64, inputShape = intArrayOf(2), activation = Activations.Relu),
        Dense(32, activation = Activations.Relu),
        Dense(1, activation = Activations.Linear)
    )

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MSE
        )

        it.fit(dataset, epochs = 100, batchSize = 10)
        val prediction = it.predict(floatArrayOf(1500f, 3f))
        println("Predicted price for a house of 1500 sq ft with 3 bedrooms: $prediction")
    }
}
