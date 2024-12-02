(ns mnist-classifier.core
  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.layers DenseLayer OutputLayer]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.evaluation.classification Evaluation]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.learning.config Adam]
           [org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction]))

(defn -main [& args]
  ;; Hyperparameters
  (let [batch-size 64
        output-num 10
        rng-seed 123
        num-epochs 10
        learning-rate 0.001]

    ;; Load MNIST data
    (def train-iter (MnistDataSetIterator. batch-size true rng-seed))
    (def test-iter (MnistDataSetIterator. batch-size false rng-seed))

    ;; Build neural network configuration
    (def nn-config (-> (NeuralNetConfiguration$Builder.)
                       (.seed rng-seed)
                       (.updater (Adam. learning-rate))
                       (.list)
                       (.layer 0 (DenseLayer/builder .nIn 784 .nOut 256 .activation Activation/RELU .build))
                       (.layer 1 (DenseLayer/builder .nIn 256 .nOut 256 .activation Activation/RELU .build))
                       (.layer 2 (OutputLayer/builder .nIn 256 .nOut output-num .activation Activation/SOFTMAX .lossFunction LossFunctions$LossFunction/MCXENT .build))
                       (.build)))

    ;; Initialize the model
    (def model (MultiLayerNetwork. nn-config))
    (.init model)
    (.setListeners model (ScoreIterationListener. 10))

    ;; Train the model
    (dotimes [i num-epochs]
      (println "Epoch" (inc i))
      (.fit model train-iter))

    ;; Evaluate the model
    (def eval (Evaluation. output-num))
    (while (.hasNext test-iter)
      (let [ds (.next test-iter)
            output (.output model (.getFeatures ds))]
        (.eval eval (.getLabels ds) output)))

    (println (.stats eval))))
