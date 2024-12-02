import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

fun main() {
    // Example data: house sizes and prices
    val houseSizes = floatArrayOf(1400f, 1600f, 1700f, 1875f, 1100f, 1550f, 5f, 1700f)
    val housePrices = floatArrayOf(245000f, 312000f, 279000f, 308000f, 199000f, 219000f, 405000f, 324000f, 319000f, 299000f)

    val dataset = OnHeapDataset.create(arrayOf(houseSizes), housePrices)

    val model = Sequential.of(
        Dense(1, inputShape = intArrayOf(1), activation = Activations.Linear)
    )

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MSE
        )

        it.fit(dataset, epochs = 100, batchSize = 10)
        val prediction = it.predict(floatArrayOf(1500f))
        println("Predicted price for a house of 1500 sq ft: $prediction")
    }
}
