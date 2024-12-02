{-# LANGUAGE OverloadedStrings #-}

import TensorFlow.Core
import TensorFlow.Minimize
import TensorFlow.Ops
import TensorFlow.Variable (initializedVariable)
import TensorFlow.Types (TensorData(..))
import Data.Vector (fromList)

-- Define the model
createModel :: MonadBuild m => TensorData Float -> m (Tensor Build Float)
createModel input = do
    let layer1 = relu (matMul input weights1 `add` biases1)
        layer2 = relu (matMul layer1 weights2 `add` biases2)
    return (matMul layer2 weightsOut `add` biasesOut)
  where
    weights1 = initializedVariable (fromList [10, 64])
    biases1 = initializedVariable (fromList [64])
    weights2 = initializedVariable (fromList [64, 64])
    biases2 = initializedVariable (fromList [64])
    weightsOut = initializedVariable (fromList [64, 1])
    biasesOut = initializedVariable (fromList [1])

-- Define the loss function
meanSquaredError :: MonadBuild m => Tensor Build Float -> Tensor Build Float -> m (Tensor Build Float)
meanSquaredError predictions targets = do
    let diff = predictions `sub` targets
    return (mean diff)

-- Training the model
trainModel :: TensorData Float -> TensorData Float -> SessionT IO ()
trainModel inputData targetData = do
    (predictions, loss, trainOp) <- build $ do
        input <- placeholder [None, 10]
        target <- placeholder [None, 1]
        output <- createModel input
        loss <- meanSquaredError output target
        trainOp <- minimizeWith defaultGradientDescent loss
        return (output, loss, trainOp)
    forM_ [1..10] $ \_ -> do
        _ <- runWithFeeds [feed input inputData, feed target targetData] trainOp
        return ()

main :: IO ()
main = runSession $ do
    let xTrain = TensorData (fromList [100, 10]) (replicate 1000 0.5 :: [Float])
        yTrain = TensorData (fromList [100, 1]) (replicate 100 0.5 :: [Float])
    trainModel xTrain yTrain
