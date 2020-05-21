# Backpropagation and Gradient Descent

Now, in this repository, similar to [simple backward NN](https://github.com/majidhamzavi/Feed-Backward-NN), we can implement the backpropagation calculations for a NN with 2 hidden layers. It is up to you to add more layers, but calculations could be tedious. 

These are the flow for a NN with 2 hidden layers: 

   1- <img src="https://render.githubusercontent.com/render/math?math=H1 = \textit{Relu}(x * w0 %2B B0)">


   2- <img src="https://render.githubusercontent.com/render/math?math=H2 = \textit{Relu}(H1 * w1 %2B B1)">


   3- <img src="https://render.githubusercontent.com/render/math?math=Out = \textit{Sigmoid}(H2 * w2 %2B B2)">
    
Once we randomly initialized weights and biases, we go back to the first layer and update the weights and biases by minimizing the loss function as explained in [my previous repository](https://github.com/majidhamzavi/Feed-Backward-NN) with respect to <img src="https://render.githubusercontent.com/render/math?math=w0, w1, w2, B0, B1,  and  B2">. It is easy to calculate the gradient errors for each layer and then update weights and biases. Once updating weights and biases, we choose a learning rate to make sure to achieve a good performance. You can read more on the learning rate in this [article](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/).

