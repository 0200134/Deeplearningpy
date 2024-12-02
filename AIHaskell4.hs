{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Control.Exception (catch, IOException)
import Data.Csv (decode, HasHeader(..))
import qualified Data.ByteString.Lazy as BL
import Numeric.LinearAlgebra
import Torch
import Torch.Functional as F
import Torch.NN
import GHC.Generics (Generic)

-- Define the neural network
data MLP = MLP { fc1 :: Linear, fc2 :: Linear, fc3 :: Linear } deriving (Generic, Show)

instance Parameterized MLP

instance Randomizable (LinearSpec -> LinearSpec -> LinearSpec -> MLP) MLP where
  sample (LinearSpec inDim hiddenDim outDim) = MLP
    <$> sample (LinearSpec inDim hiddenDim)
    <*> sample (LinearSpec hiddenDim hiddenDim)
    <*> sample (LinearSpec hiddenDim outDim)

-- Define the forward pass
forward :: MLP -> Tensor -> Tensor
forward MLP{..} = relu . linear fc3 . relu . linear fc2 . relu . linear fc1

-- Load and preprocess the MNIST dataset
loadMNIST :: FilePath -> IO (Either String (Matrix Float, Vector Float))
loadMNIST path = do
  csvData <- BL.readFile path
  let decoded = decode HasHeader csvData
  return $ case decoded of
    Left err -> Left err
    Right rows -> Right (fromLists (map init rows), fromList (map last rows))

-- Exception handler
handler :: IOException -> IO (Either String (Matrix Float, Vector Float))
handler ex = do
  putStrLn $ "An error occurred: " ++ show ex
  return $ Left (show ex)

-- Train the model
train :: MLP -> Tensor -> Tensor -> Optimizer -> IO MLP
train model inputs targets optimizer = do
  let forwardPass = forward model inputs
      loss = F.mseLoss forwardPass targets
  runStep model optimizer loss
  return model

-- Main function
main :: IO ()
main = do
  result <- (loadMNIST "mnist_train.csv") `catch` handler
  case result of
    Left err -> putStrLn $ "Failed to load dataset: " ++ err
    Right (inputs, targets) -> do
      let inputsTensor = asTensor inputs
          targetsTensor = asTensor targets
      initialModel <- sample (LinearSpec 784 128 10)
      let optimizer = GD (initialModel, defaultSGDOpts 0.01)
      trainedModel <- foldM (\model _ -> train model inputsTensor targetsTensor optimizer) initialModel [1..10]
      putStrLn "Training completed."
