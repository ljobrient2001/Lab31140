#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as numpy
import scipy.stats as stat
import scipy.optimize as sp


# In[2]:


period = numpy.array([2.739, 2.742, 2.741, 2.744, 2.746, 2.803, 2.749, 2.746, 2.749, 2.746, 2.746]) 
time = numpy.array([0,10,20,30,40,50,60,70,80,90,100])
amplitude = numpy.array([8.7,8.23,8.08,7.79,7.55,7.31,7.11,7.00,6.72,6.53, 6.46]) 
masses = numpy.array([0.298,1.1]) 


# In[3]:


plt.plot(time,period, 'ro') 

plt.ylabel('period')
plt.xlabel('mass')
plt.title('Period as a function of mass')


# In[4]:


plt.plot(time,amplitude, 'ro') 

plt.ylabel('period')
plt.xlabel('amplitude')
plt.title('Period as a function of length')


# In[5]:


x = numpy.arange(0,4*numpy.pi,0.1)   # start,stop,step
y = numpy.sin(A*x)
plt.plot(x,y)
plt.show()


# In[44]:


from scipy.optimize import curve_fit
fit1 = sp.curve_fit(time, period,1) #1 refers to linear here.
p = sp.curve_fit(fit1) #Do this to convert it into a usable form 


# In[45]:


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines

ax.plot(time, period, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(time, p(time), c = 'r', ls = '--', label= p) #plotting the linear formula

plt.legend(loc=2) #Legends are very useful

plt.draw()
plt.title("Period versus Time")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Period(s)')
print(p) 
print(fit1)


# In[46]:


fit1 = numpy.polyfit(time, amplitude, 1) #1 refers to linear here.
p = numpy.poly1d(fit1) #Do this to convert it into a usable form 


# In[47]:


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines

ax.plot(time, amplitude, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(time, p(time), c = 'r', ls = '--', label= p) #plotting the linear formula

plt.legend(loc=2) #Legends are very useful
plt.draw()
plt.title("Period versus Time")
ax.set_xlabel('Time (m)')
ax.set_ylabel('Period(s)')
print(p) 
print(fit1)


# In[48]:


from scipy.optimize import curve_fit


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines

ax.plot(time, amplitude, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(time, p(time), c = 'r', ls = '--', label= p) #plotting the linear formula
A=8.74482
B=2.75
C=3
D=28.22
Amp=A*numpy.exp(-B*x)*numpy.cos(C*x+D)
xdata = numpy.linspace(0,10,20,30,40,50)
V=curve_fit(xdata, Amp(xdata))
plt.plot(V)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(x, a, b, c, d):
    return a*numpy.exp(-b*x)*numpy.cos(c*x+d)

xdata = numpy.linspace(0, 4, 50)
y = func(xdata, 8.74428, 3, 2.75, 28.22)
numpy.random.seed(1729)
y_noise = 0.2 * numpy.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)

plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[6]:


from numpy import sin
from numpy import sqrt
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot


# define the true objective function
def objective(x, a, b, c, d, e):
	return a*numpy.exp(-b*x)*numpy.cos(c*x+d) + e

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header=None)
data = dataframe.values

x = numpy.array([0,10,20,30,40,50,60,70,80,90,100])
y=numpy.array([8.7,8.23,8.08,7.79,7.55,7.31,7.11,7.00,6.72,6.53, 6.46]) 
popt, _ = curve_fit(objective, x, y)
a, b, c, d, e = popt
print(popt)

# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e)
# create a line plot for the mapping function

fit1 = numpy.polyfit(time, amplitude, 1) #1 refers to linear here.
p = numpy.poly1d(fit1) #Do this to convert it into a usable form 


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines



ax.plot(time, amplitude, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(time, p(time), c = 'r', ls = '--', label= p) #plotting the linear formula
plt.draw()
plt.legend(loc=2) #Legends are very useful
plt.title("amplitude versus Time")
ax.set_xlabel('Time (m)')
ax.set_ylabel('amplitude(s)')
print(p) 
print(fit1)

pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()


# In[43]:


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c, d, e):
    return a*numpy.exp(-b*x)*numpy.cos(c*x+d) + e

xdata = numpy.array([0,10,20,30,40,50,60,70,80,90,100])
ydata = numpy.array([8.7,8.23,8.08,7.79,7.55,7.31,7.11,7.00,6.72,6.53, 6.46]) 

plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-',
    label='fit: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f, e=%5.5f' % tuple(popt))

popt, pcov = curve_fit(func, xdata, ydata, bounds=(6.5, [8.7]))

plt.plot(xdata, func(xdata, *popt), 'g--',
    label='fit: a=%5.5f, b=%5.5f, c=%5.5f, d=%5.5f, e=%5.5f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[12]:


from numpy import sin
from numpy import sqrt
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot


# define the true objective function
def objective(x, a, b, c, d, e):
	return a*numpy.exp(-b*x)*numpy.cos(c*x+d) + e

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header=None)
data = dataframe.values

x = numpy.array([0,10,20,30,40,50,60,70,80,90])
y = numpy.array([3.62,3.51,3.38,3.28,3.23,3.13,3.13,3.097,2.94,2.91]) 
popt, _ = curve_fit(objective, x, y)
a, b, c, d, e = popt
print(popt)

# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e)
# create a line plot for the mapping function

fit1 = numpy.polyfit(x, y, 1) #1 refers to linear here.
p = numpy.poly1d(fit1) #Do this to convert it into a usable form 


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines



ax.plot(x, y, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(x, p(x), c = 'r', ls = '--', label= p) #plotting the linear formula
plt.draw()
plt.legend(loc=2) #Legends are very useful
plt.title("amplitude versus Time")
ax.set_xlabel('Time (m)')
ax.set_ylabel('amplitude(s)')
print(p) 
print(fit1)

pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()


# In[12]:


from numpy import sin
from numpy import sqrt
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot


# define the true objective function
def objective(x, a, b, c, d, e):
	return a*numpy.exp(-b*x)*numpy.cos(c*x+d) + e

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header=None)
data = dataframe.values

x = numpy.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130])
y = numpy.array([3.42,2.68,2.66,2.11,1.90,1.56,1.25,1.25,1.01,0.81,0.74,0.60,0.44,0.37]) 
popt, _ = curve_fit(objective, x, y)
a, b, c, d, e = popt
print(popt)

# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e)
# create a line plot for the mapping function

fit1 = numpy.polyfit(x, y, 1) #1 refers to linear here.
p = numpy.poly1d(fit1) #Do this to convert it into a usable form 


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines



ax.plot(x, y, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(x, p(x), c = 'r', ls = '--', label= p) #plotting the linear formula
plt.draw()
plt.legend(loc=2) #Legends are very useful
plt.title("amplitude versus Time")
ax.set_xlabel('Time (m)')
ax.set_ylabel('amplitude(s)')
print(p) 
print(fit1)

pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()


# In[7]:


from numpy import sin
from numpy import sqrt
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot


# define the true objective function
def objective(x, a, b, c, d, e):
	return a*numpy.exp(-b*x)*numpy.cos(c*x+d) + e

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header=None)
data = dataframe.values

x = numpy.array([0,10,20,30,40,50,60,70,80,90])
y = numpy.array([3.06,2.75,2.65,2.55,2.50,2.35,2.31,2.23,2.19,2.11]) 
popt, _ = curve_fit(objective, x, y)
a, b, c, d, e = popt
print(popt)

# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e)
# create a line plot for the mapping function

fit1 = numpy.polyfit(x, y, 1) #1 refers to linear here.
p = numpy.poly1d(fit1) #Do this to convert it into a usable form 


fig=plt.figure() 
ax=fig.add_subplot(111) #So we can add multiple lines



ax.plot(x, y, c='b',marker="^",ls='--',label='y',fillstyle='none') #Data is horizontial axis, then vertical axis
ax.plot(x, p(x), c = 'r', ls = '--', label= p) #plotting the linear formula
plt.draw()
plt.legend(loc=2) #Legends are very useful
plt.title("amplitude versus Time")
ax.set_xlabel('Time (m)')
ax.set_ylabel('amplitude(s)')
print(p) 
print(fit1)

pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()

