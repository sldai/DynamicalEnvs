# kinodynamic planning environments 

A set of environments designed for kinodynamic planning. And the environments are wrapped in OpenAI [gym](https://github.com/openai/gym), so you can also use the reinforcement learning for planning and control.

## Cart pole
Dynamic: https://arxiv.org/pdf/1405.2872.pdf

## Dubin vehicle

Kinematic: 

![](https://latex.codecogs.com/png.latex?%5Cbegin%7Bbmatrix%7D%20%5Cdot%7Bx%7D%5C%5C%20%5Cdot%7By%7D%5C%5C%20%5Cdot%7B%5Ctheta%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20v%5Ccos%28%5Ctheta%29%5C%5C%20v%5Csin%28%5Ctheta%29%5C%5C%20v/%28d/%5Ctan%28%5Cdelta%29%29%20%5Cend%7Bbmatrix%7D)

For the reinforcement use

Observation:

![observation](https://latex.codecogs.com/png.latex?%5B%5Chat%7Bx%7D%2C%20%5Chat%7By%7D%2C%20%5Ctheta_s%2C%20%5Ctheta_g%5D%2C%20%5Chat%7Bx%7D%20%3D%20x_s%20-%20x_g)

Action:

![action](https://latex.codecogs.com/png.latex?%5Bv%2C%20%5Cdelta%5D)


## Quadcopter

Dynamc: https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf

For the reinforcement use

Observation:

![observation](https://latex.codecogs.com/png.latex?%5B%5Chat%7Bx%7D%2C%20%5Chat%7By%7D%2C%20%5Chat%7Bz%7D%2C%20%5Cdot%7Bx%7D%2C%20%5Cdot%7By%7D%2C%20%5Cdot%7Bz%7D%2C%20%5Cphi%2C%20%5Ctheta%20%2C%5Cpsi%20%2C%20%5Cdot%7B%5Cphi%7D%2C%20%5Cdot%7B%5Ctheta%7D%20%2C%5Cdot%7B%5Cpsi%7D%20%5D)

Action:

![action](https://latex.codecogs.com/png.latex?%5Bf%2C%20r_x%2C%20r_y%2C%20r_z%5D)

![quadcopter](quadcopter.gif)
