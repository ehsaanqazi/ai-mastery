This quiz covers the whole Calculus module: functions, limits, derivatives, partial derivatives, the chain rule, gradients, gradient descent and optimisation. Answer every question to see your score. Your best result is saved. Aim for at least 13 out of 16 before moving on to Probability and Statistics.

```quiz
[
  {
    "q": "In machine learning, the loss is usually written L(w). Which quantities are held fixed while you vary w?",
    "options": ["The weights", "The training data", "The gradient", "The learning rate and the weights"],
    "answer": 1,
    "why": "During training the data are constants and the weights are the variables, so the loss is a function of the weights."
  },
  {
    "q": "What is the limit of (x^2 - 16)/(x - 4) as x approaches 4?",
    "options": ["0", "4", "It does not exist", "8"],
    "answer": 3,
    "why": "For x not equal to 4 the expression simplifies to x + 4, which approaches 8."
  },
  {
    "q": "Which statement about the derivative f'(x) is correct?",
    "options": [
      "It is the slope of the tangent line at x, the limit of rise over run as the step shrinks to zero",
      "It is the area under the curve up to x",
      "It is the value of f at x",
      "It is always positive for a minimum"
    ],
    "answer": 0,
    "why": "The derivative is the instantaneous rate of change: the limit of the secant slope as h goes to 0."
  },
  {
    "q": "What is the derivative of f(x) = 3x^2 + 2x - 5?",
    "options": ["3x + 2", "6x + 2", "6x - 5", "6x^2 + 2"],
    "answer": 1,
    "why": "Power rule on each term: 3 times 2x is 6x, 2x gives 2, and the constant -5 gives 0."
  },
  {
    "q": "With `f = lambda x: x**2` and `h = 1e-5`, what does `print(round((f(3 + h) - f(3 - h)) / (2 * h), 4))` print?",
    "options": ["9.0", "3.0", "6.0", "0.0"],
    "answer": 2,
    "why": "It is the central difference estimate of the derivative of x^2 at 3, which is 2 times 3 = 6."
  },
  {
    "q": "Why should you not use h = 1e-17 in a numerical derivative with 64-bit floats?",
    "options": [
      "Python raises an error for numbers that small",
      "x + h rounds to x, so the difference in the numerator becomes zero or meaningless",
      "The derivative becomes infinite",
      "Smaller h is always more accurate, so 1e-17 is actually ideal"
    ],
    "answer": 1,
    "why": "Floats keep about 16 significant digits. Subtracting nearly equal numbers loses precision, and below a point the step disappears entirely."
  },
  {
    "q": "For f(x, y) = x^3 y + 2y, what is the partial derivative with respect to y?",
    "options": ["x^3 + 2", "3x^2 y", "3x^2 + 2", "x^3 y"],
    "answer": 0,
    "why": "Treat x as a constant: x^3 y differentiates to x^3 and 2y to 2."
  },
  {
    "q": "For the MSE loss of y_hat = wx + b, the partial derivative with respect to w is (2/n) times the sum of r_i x_i, where r_i is the residual. What does this tell you?",
    "options": [
      "The weight ignores examples with large inputs",
      "The bias and weight always receive the same update",
      "Examples with larger inputs x_i have more influence on how w should change",
      "The derivative does not depend on the data"
    ],
    "answer": 2,
    "why": "Each residual is weighted by its input x_i, so examples with large inputs contribute more to the weight's partial derivative."
  },
  {
    "q": "Using the chain rule, what is the derivative of y = e^(3x)?",
    "options": ["e^(3x)", "3e^x", "e^(3x) / 3", "3e^(3x)"],
    "answer": 3,
    "why": "Outer derivative e^u evaluated at u = 3x, times inner derivative 3."
  },
  {
    "q": "In a computational graph, a variable feeds into the output along two different paths. How is its total derivative computed?",
    "options": [
      "Multiply local derivatives along each path, then add the paths together",
      "Multiply the two paths together",
      "Use only the shorter path",
      "Average the local derivatives of all nodes"
    ],
    "answer": 0,
    "why": "The multivariable chain rule: products along paths, sums across paths. Backpropagation organises this calculation efficiently."
  },
  {
    "q": "A 20-layer network uses sigmoid activations everywhere and its early layers barely learn. Which explanation fits best?",
    "options": [
      "The learning rate is too high",
      "The chain rule multiplies many sigmoid slopes, each at most 0.25, so early-layer gradients vanish",
      "Sigmoid is not differentiable",
      "The network has too few parameters"
    ],
    "answer": 1,
    "why": "Each layer contributes a factor no larger than 0.25 to the backward product, which shrinks rapidly with depth."
  },
  {
    "q": "At a point, the gradient of the loss is (0, -4, 3). What is the steepest possible rate of increase of the loss from there?",
    "options": ["7", "1", "5", "-4"],
    "answer": 2,
    "why": "The steepest slope equals the length of the gradient: the square root of 0 + 16 + 9, which is 5."
  },
  {
    "q": "For L(w) = (w - 3)^2, gradient descent starts at w = 1 with learning rate 0.25. What is w after one step?",
    "options": ["2.0", "1.5", "2.5", "3.0"],
    "answer": 0,
    "why": "The gradient is 2(1 - 3) = -4. The update is 1 - 0.25 times -4 = 2."
  },
  {
    "q": "With L(w) = (w - 3)^2, which learning rate makes gradient descent diverge?",
    "options": ["0.05", "0.5", "0.9", "1.2"],
    "answer": 3,
    "why": "The gap to 3 is multiplied by 1 - 2 times lr each step. For lr = 1.2 the multiplier is -1.4, whose size exceeds 1, so the gap grows."
  },
  {
    "q": "At a critical point the Hessian has eigenvalues 4 and -1. What kind of point is it?",
    "options": ["Local minimum", "Saddle point", "Local maximum", "Global minimum"],
    "answer": 1,
    "why": "Mixed-sign eigenvalues mean the surface curves up in one direction and down in another: a saddle point."
  },
  {
    "q": "Which statement about convexity is correct?",
    "options": [
      "Neural network losses are convex, so any minimum is the global one",
      "A convex function can have several local minima with different values",
      "Linear regression with MSE is convex, so gradient descent with a suitable learning rate reaches the global minimum",
      "Convexity means the gradient is zero everywhere"
    ],
    "answer": 2,
    "why": "For a convex function every local minimum is global. Linear regression is convex; neural networks are not."
  }
]
```
