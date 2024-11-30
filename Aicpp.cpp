#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/tensor.h>

int main() {
  // Create a TensorFlow session
  tensorflow::Session* session = tensorflow::Session::NewSession(tensorflow::SessionOptions());

  // Define the graph
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto x = tensorflow::ops::Placeholder(root, tensorflow::DataType::DT_FLOAT);
  auto W = tensorflow::ops::Variable(root, {2, 1}, tensorflow::DataType::DT_FLOAT);
  auto b = tensorflow::ops::Variable(root, {1}, tensorflow::DataType::DT_FLOAT);
  auto y = tensorflow::ops::MatMul(root, x, W) + b;

  // Define the loss function
  auto y_true = tensorflow::ops::Placeholder(root, tensorflow::DataType::DT_FLOAT);
  auto loss = tensorflow::ops::Square(root, y - y_true);
  auto loss_mean = tensorflow::ops::ReduceMean(root, loss, {0});

  // Define the optimizer
  auto optimizer = tensorflow::ops::AdamOptimizer(root, 0.01);
  auto train_op = optimizer.Minimize(loss_mean);

  // Initialize variables
  tensorflow::Tensor x_val({2, 1}, {1.0f, 2.0f});
  tensorflow::Tensor y_val({1}, {3.0f});
  session->Run({{W, b}}, {}, {});

  // Training loop
  for (int i = 0; i < 1000; ++i) {
    std::vector<std::pair<tensorflow::Tensor, tensorflow::Tensor>> inputs = {{x, x_val}, {y_true, y_val}};
    std::vector<tensorflow::Tensor> outputs;
    session->Run(inputs, {train_op}, &outputs);
  }

  // Make a prediction
  tensorflow::Tensor prediction;
  session->Run({{x, x_val}}, {y}, &prediction);
  std::cout << prediction.flat<float>()(0) << std::endl;

  // Clean up
  session->Close();
  delete session;
  return 0;
}
