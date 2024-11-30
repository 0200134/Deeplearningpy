module neural_network
    implicit none
    real, parameter :: eta = 0.01
    integer, parameter :: input_neurons = 3, hidden_neurons = 3, output_neurons = 1
    real :: weights_ih(hidden_neurons, input_neurons)
    real :: weights_ho(output_neurons, hidden_neurons)
    real :: bias_h(hidden_neurons)
    real :: bias_o(output_neurons)
contains
    subroutine initialize_weights()
        call random_number(weights_ih)
        call random_number(weights_ho)
        call random_number(bias_h)
        call random_number(bias_o)
    end subroutine initialize_weights

    function sigmoid(x)
        real :: x, sigmoid
        sigmoid = 1.0 / (1.0 + exp(-x))
    end function sigmoid

    function feedforward(inputs)
        real :: inputs(input_neurons), hidden(hidden_neurons), output(output_neurons)
        feedforward = sigmoid(matmul(weights_ho, sigmoid(matmul(weights_ih, inputs) + bias_h)) + bias_o)
    end function feedforward

    subroutine backpropagation(inputs, targets)
        real :: inputs(input_neurons), targets(output_neurons)
        real :: hidden(hidden_neurons), output(output_neurons)
        real :: error_o(output_neurons), error_h(hidden_neurons)

        hidden = sigmoid(matmul(weights_ih, inputs) + bias_h)
        output = sigmoid(matmul(weights_ho, hidden) + bias_o)

        error_o = targets - output
        error_h = matmul(transpose(weights_ho), error_o)

        weights_ho = weights_ho + eta * matmul(error_o * output * (1 - output), transpose(hidden))
        weights_ih = weights_ih + eta * matmul(error_h * hidden * (1 - hidden), transpose(inputs))
        bias_o = bias_o + eta * error_o * output * (1 - output)
        bias_h = bias_h + eta * error_h * hidden * (1 - hidden)
    end subroutine backpropagation
end module neural_network

program main
    use neural_network
    implicit none
    real :: inputs(input_neurons), targets(output_neurons)
    call initialize_weights()

    ! Training example
    inputs = [1.0, 0.0, 1.0]
    targets = [1.0]
    call backpropagation(inputs, targets)
    print *, "Output: ", feedforward(inputs)
end program main
