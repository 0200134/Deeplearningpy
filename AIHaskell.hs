import Numeric.LinearAlgebra
import System.Random

-- Define the activation function (Sigmoid function)
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

-- Derivative of the sigmoid function
sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- Generate random weights
initWeights :: Int -> Int -> IO (Matrix Double)
initWeights rows cols = randn rows cols

-- Forward propagation
forward :: Matrix Double -> Matrix Double -> Vector Double -> (Matrix Double, Vector Double)
forward input weights bias = (z, a)
  where
    z = (input `mult` weights) + (asColumn bias)
    a = cmap sigmoid z

-- Backward propagation (gradient descent)
backward :: Matrix Double -> Matrix Double -> Vector Double -> Matrix Double -> (Matrix Double, Vector Double)
backward input weights bias delta = (weights', bias')
  where
    gradient = (trans input) `mult` delta
    weights' = weights - (0.1 * gradient)
    bias' = bias - (0.1 * (sumElements delta))

-- Train the neural network
train :: Matrix Double -> Matrix Double -> Vector Double -> IO (Matrix Double, Vector Double)
train input targets weights bias = do
  let (z, a) = forward input weights bias
      delta = (a - targets) * cmap sigmoid' z
  return $ backward input weights bias delta

-- Main function to initialize and train the network
main :: IO ()
main = do
  let input = (3><2) [0.5, 1.5, 1.0, 2.0, 1.5, 3.0]
      targets = vector [0.0, 1.0, 1.0]
  weights <- initWeights 2 1
  let bias = vector [0.0]
  (trainedWeights, trainedBias) <- train input targets weights bias
  putStrLn "Trained Weights:"
  print trainedWeights
  putStrLn "Trained Bias:"
  print trainedBias
