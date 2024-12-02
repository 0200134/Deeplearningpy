import org.tensorflow.keras.models.Sequential
import org.tensorflow.keras.layers.Dense
import org.tensorflow.keras.layers.LeakyReLU
import org.tensorflow.keras.optimizers.Adam
import org.tensorflow.keras.datasets.mnist
import org.tensorflow.keras.losses.BinaryCrossentropy
import org.tensorflow.keras.backend.reshape
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32

// Define the generator model
fun buildGenerator(noiseDim: Int): Sequential {
    val model = Sequential()
    model.add(Dense(256, inputShape = intArrayOf(noiseDim)))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(28 * 28 * 1, activation = "tanh"))
    return model
}

// Define the discriminator model
fun buildDiscriminator(): Sequential {
    val model = Sequential()
    model.add(Dense(1024, inputShape = intArrayOf(28 * 28 * 1)))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2f))
    model.add(Dense(1, activation = "sigmoid"))
    return model
}

// Training the GAN
fun trainGAN(epochs: Int, batchSize: Int, noiseDim: Int) {
    val (trainImages, _) = mnist.load_data()
    val trainImagesNormalized = trainImages.reshape(Shape.of(60000, 28 * 28)).div(255.0).sub(0.5).mul(2)

    val generator = buildGenerator(noiseDim)
    val discriminator = buildDiscriminator()
    discriminator.compile(loss = BinaryCrossentropy(), optimizer = Adam(0.0002f, 0.5f))

    val gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = false
    gan.compile(loss = BinaryCrossentropy(), optimizer = Adam(0.0002f, 0.5f))

    for (epoch in 0 until epochs) {
        for (i in 0 until (trainImagesNormalized.shape().size(0) / batchSize).toInt()) {
            val noise = TFloat32.tensorOf(Shape.of(batchSize.toLong(), noiseDim.toLong())).randomNormal()
            val generatedImages = generator.predict(arrayOf(noise))

            val realImages = trainImagesNormalized.slice(i * batchSize, (i + 1) * batchSize)
            val realLabels = TFloat32.tensorOf(Shape.of(batchSize.toLong(), 1)).fill(1.0f)
            val fakeLabels = TFloat32.tensorOf(Shape.of(batchSize.toLong(), 1)).fill(0.0f)

            val dLossReal = discriminator.trainOnBatch(arrayOf(realImages), arrayOf(realLabels))
            val dLossFake = discriminator.trainOnBatch(arrayOf(generatedImages), arrayOf(fakeLabels))
            val dLoss = 0.5 * (dLossReal + dLossFake)

            val gLoss = gan.trainOnBatch(arrayOf(noise), arrayOf(realLabels))

            if (i % 10 == 0) {
                println("Epoch $epoch, Batch $i, D Loss: $dLoss, G Loss: $gLoss")
            }
        }
    }
}

fun main() {
    val noiseDim = 100
    val epochs = 10000
    val batchSize = 64
    trainGAN(epochs, batchSize, noiseDim)
}
