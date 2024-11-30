module neural_network
  implicit none
  integer, parameter :: input_size = 3072 ! CIFAR-10 images are 32x32x3 = 3072
  integer, parameter :: hidden_size = 128
  integer, parameter :: output_size = 10
  real, dimension(input_size, hidden_size) :: weights1
  real, dimension(hidden_size, output_size) :: weights2
  real, dimension(hidden_size) :: bias1
  real, dimension(output_size) :: bias2

contains

  subroutine initialize()
    call random_number(weights1)
    call random_number(weights2)
    call random_number(bias1)
    call random_number(bias2)
  end subroutine initialize

  function relu(x) result(y)
    real, intent(in) :: x
    real :: y
    if (x > 0) then
      y = x
    else
      y = 0.0
    end if
  end function relu

  function softmax(x) result(y)
    real, dimension(:), intent(in) :: x
    real, dimension(size(x)) :: y
    real :: sum_exp
    integer :: i
    sum_exp = sum(exp(x))
    do i = 1, size(x)
      y(i) = exp(x(i)) / sum_exp
    end do
  end function softmax

  function forward(input) result(output)
    real, dimension(input_size), intent(in) :: input
    real, dimension(output_size) :: output
    real, dimension(hidden_size) :: hidden
    hidden = matmul(input, weights1) + bias1
    hidden = reshape([ (relu(hidden(i)), i=1, size(hidden)) ], shape(hidden))
    output = matmul(hidden, weights2) + bias2
    output = softmax(output)
  end function forward

end module neural_network

program main
  use neural_network
  implicit none
  real, dimension(input_size) :: input
  real, dimension(output_size) :: output

  call initialize()

  ! Load your data into the input array here
  ! Example: input = ...

  output = forward(input)

  print *, "Output: ", output

end program main
