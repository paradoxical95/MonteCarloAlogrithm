Open terminal/cmd if you have CUDA Toolkit or NVCC installed.

To run 
-> _**nvcc MonteCarlo.cu -o MonteCarlo.exe**_  (Windows)
OR
-> _**nvcc MonteCarlo.cu -o MonteCarlo**_  (Linux)

Followed by
-> _**./MonteCarlo**_ (Linux) 
OR 
-> _**MonteCarlo.exe**_ (Windows)

The Monte Carlo method is a statistical technique that allows us to estimate numerical values through random sampling. One of its classic applications is estimating the value of π (pi) using a geometric approach.


Why Monte Carlo?

  Simplicity: The method is straightforward to implement, especially with a large number of points, making it ideal for numerical approximations.
  
  Parallelization: Monte Carlo simulations lend themselves well to parallel computing, as each random point generation and its evaluation is independent of others. This makes it suitable for GPU implementations.
  
  Versatility: The same principle can be applied to a variety of problems, not just estimating π. It's used in finance, physics, engineering, and more for problems involving uncertainty.
