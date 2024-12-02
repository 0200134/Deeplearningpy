#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/tensor.h>

#include <iostream>
#include <fstream>

using namespace tensorflow;

int main() {
  // Load the TensorFlow model
  SessionOptions options;
  Session* session;
  Status status = NewSession(options, &session);
  if (!status.ok()) {
    std::cerr << "Could not create session: " << status.ToString() << std::endl;
    return -1;
  }

  // Load the graph and labels
  std::string graph_path = "model.pb";
  std::string labels_path = "labels.txt";
  Tensor graph_tensor(DT_STRING, TensorShape());
  graph_tensor.Scalar<std::string>()() = ReadFileToStringOrDie(graph_path);
  Tensor labels_tensor(DT_STRING, TensorShape());
  labels_tensor.Scalar<std::string>()() = ReadFileToStringOrDie(labels_path);

  std::vector<std::pair<std::string, Tensor>> inputs = {
    {"graph_def", graph_tensor},
    {"labels", labels_tensor}
  };

  std::vector<Tensor> outputs;
  status = session->Run(inputs, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Could not run inference: " << status.ToString() << std::endl;
    return -1;
  }

  // Process the output tensor
  auto output_tensor = outputs[0];
  auto output_flat = output_tensor.flat<float>();
  int max_index = 0;
  float max_value = output_flat(0);
  for (int i = 1; i < output_flat.size(); ++i) {
    if (output_flat(i) > max_value) {
      max_index = i;
      max_value = output_flat(i);
    }
  }

  // Read the label from the labels file
  std::ifstream label_file(labels_path);
  std::string label;
  std::getline(label_file, label);
  for (int i = 0; i < max_index; ++i) {
    std::getline(label_file, label);
  }

  std::cout << "Predicted class: " << label << std::endl;

  session->Close();
  delete session;
  return 0;
}
