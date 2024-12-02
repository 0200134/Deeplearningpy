{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Control.Exception (catch, IOException)
import Data.Aeson (FromJSON, decode)
import Data.Maybe (fromJust)
import qualified Data.ByteString.Lazy as BL
import System.Environment (getEnv)
import System.IO (hPutStrLn, stderr)
import System.Log.Logger (Priority(..), updateGlobalLogger, rootLoggerName, setHandlers, setLevel, infoM, errorM)
import System.Log.Handler.Simple (streamHandler)
import System.Log.Handler (setFormatter)
import System.Log.Formatter (simpleLogFormatter)
import GHC.Generics (Generic)
import Numeric.LinearAlgebra
import System.Random

data Config = Config {
  learningRate :: Double,
  epochs :: Int
} deriving (Generic, Show)

instance FromJSON Config

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

-- Load and preprocess the dataset
loadData :: FilePath -> IO (Either String (Matrix Float, Vector Float))
loadData path = do
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
train :: MLP -> Tensor -> Tensor -> Optimizer -> Int -> IO MLP
train model inputs targets optimizer epochs = foldM (\m _ -> do
    let forwardPass = forward m inputs
        loss = mseLoss forwardPass targets
    runStep m optimizer loss
    infoM "MLP.Train" "Completed an epoch"
    return m
  ) model [1..epochs]

-- Load configuration
loadConfig :: FilePath -> IO (Either String Config)
loadConfig path = do
  jsonData <- BL.readFile path
  let config = decode jsonData
  return $ case config of
    Nothing -> Left "Failed to parse config file"
    Just cfg -> Right cfg

-- Main function
main :: IO ()
main = do
  -- Initialize logger
  handler <- streamHandler stderr INFO >>= \h -> return $
    setFormatter h (simpleLogFormatter "[$time : $prio] $msg")
  updateGlobalLogger rootLoggerName (setLevel INFO . setHandlers [handler])

  -- Load configuration
  configResult <- (loadConfig "config.json") `catch` (\e -> do
    hPutStrLn stderr ("Error loading config: " ++ show (e :: IOException))
    return $ Left "Failed to load config"
    )
  config <- case configResult of
    Left err -> do
      errorM "MLP.Config" err
      error "Exiting due to config error"
    Right cfg -> return cfg

  infoM "MLP.Main" "Starting the application"

  -- Load dataset
  datasetResult <- (loadData "dataset.csv") `catch` handler
  case datasetResult of
    Left err -> errorM "MLP.Data" ("Failed to load dataset: " ++ err)
    Right (inputs, targets) -> do
      let inputsTensor = asTensor inputs
          targetsTensor = asTensor targets
          epochs = epochs config
          lr = learningRate config
      initialModel <- sample (LinearSpec 784 128 10)
      let optimizer = GD (initialModel, defaultSGDOpts lr)
      trainedModel <- train initialModel inputsTensor targetsTensor optimizer epochs
      infoM "MLP.Main" "Training completed."
