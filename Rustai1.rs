use ndarray::Array2;
use tch::{nn, Tensor};

fn main() {
    // Create a simple neural network
    let nn = nn::Sequential::new()
        .add(nn::Linear::new(2, 10))
        .add(nn::ReLU)
        .add(nn::Linear::new(10, 1));

    // Define loss function and optimizer
    let loss = nn::criterion::MSELoss::new();
    let optimizer = nn::optim::SGD::new(nn.parameters(), 0.01);

    // Sample data
    let x = Tensor::of_slice(&[
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    .to(&tch::Device::Cpu);
    let y = Tensor::of_slice(&[2.0, 4.0, 6.0]).to(&tch::Device::Cpu);

    // Training loop
    for epoch in 0..100 {
        let y_pred = nn.forward(&x);
        let loss = loss.forward(&y_pred, &y);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        println!("Epoch {}, Loss: {}", epoch, loss.item());
    }
}
