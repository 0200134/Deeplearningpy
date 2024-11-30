use tch::{nn, Tensor};

fn main() {
    // Define the model
    let net = nn::Sequential::new()
        .add(nn::Linear::new(784, 128))
        .add(nn::ReLU)
        .add(nn::Linear::new(128, 10));

    // Define the loss function and optimizer
    let loss = nn::criterion::CrossEntropyLoss::new();
    let optimizer = nn::optim::SGD::new(net.parameters(), 0.01);

    // Load and preprocess data (e.g., MNIST dataset)
    // ...

    // Training loop
    for epoch in 0..10 {
        for (x, y) in data.iter() {
            let y_pred = net.forward(&x);
            let loss = loss.forward(&y_pred, &y);

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}
