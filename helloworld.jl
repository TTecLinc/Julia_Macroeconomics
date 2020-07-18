using PyCall
using PyPlot
@pyimport numpy as np
#default is Qt, so we cannot use it
#@pyimport matplotlib.pyplot as plt

#The first time we have to use commmand line 
x=np.linspace(0,1,100)
y=randn(100)
plt.plot(x,y,color="blue")
plt.title("Basic Graphic in PyPlot")
plt.show()
plt.figure()
plt.scatter(x,y)
