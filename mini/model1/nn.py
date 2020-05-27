import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import metrics

data=pd.read_csv("/home/karthik/Desktop/mini/model1/altered_dataset2/train.csv",header=None,skiprows=1)

df=DataFrame(data)

x=df.iloc[:, 0:3]
y_amt=df.iloc[:, 3:4]
y_waste=df.iloc[:, 4:5]
children=pd.DataFrame(df.iloc[:, 0:1])
adults=pd.DataFrame(df.iloc[:, 1:2])
elderly=pd.DataFrame(df.iloc[:, 2:3])


#for testing
data_test=pd.read_csv("/home/karthik/Desktop/mini/model1/altered_dataset2/test.csv",header=None,skiprows=1)
df_test=DataFrame(data_test)
ytest=df_test.iloc[:, 3:4]
xtest=df_test.iloc[:, 0:3]
y_waste_test=df_test.iloc[:, 4:5]


output_amt=[]
output_test=[]
wasted_amt=[]


#hyperparameters
no_neurons=5
m=x.shape[1]                                                  
n=y_amt.shape[1]                                              
no_examples=x.shape[0]
no_iter=500
alpha=0.00001


#x=np.transpose(x)

w=np.random.rand(m*no_neurons).reshape(no_neurons,m) 
w2=np.random.rand(m*no_neurons).reshape(no_neurons,m) 
b=np.ones((no_neurons,n))                                     




def eqn(x,theta,bias):
	ans=np.dot(x,theta)+bias
	#print(ans)
	return ans.reshape(x.shape[0],1)

def cost_func(m,y,pred):
	squared_error=np.square(pred-y)
	ans=np.sum(squared_error)/(2*m)
	#print(ans)
	return np.array(ans)[0]

def differential(m,y,pred,x):
	ans=(np.sum(np.dot(np.transpose(pred-y),x)))/m
	#print(ans)
	return ans


def gradient_descent(m,x,y,theta,no_iter,alpha,bias):
	ind=[]
	cost=[]
	for i in range(no_iter):
		pred=eqn(x,theta,bias)

		gradient0=differential(m,y,pred,children)
		gradient1=differential(m,y,pred,adults)
		gradient2=differential(m,y,pred,elderly)

		
		theta[0]=theta[0]-(alpha*gradient0)
		theta[1]=theta[1]-(alpha*gradient1)
		theta[2]=theta[2]-(alpha*gradient2)

				
		#print(theta,cost_func(m,y,pred))
	return theta.reshape(1,3)


for i in range(no_neurons):
	w[i]=gradient_descent(no_examples,x,y_amt,np.transpose(w[i]),no_iter,alpha,b[i])
	w2[i]=gradient_descent(no_examples,x,y_waste,np.transpose(w2[i]),no_iter,alpha,b[i])



for j in range(0,no_neurons):
	output_amt.append(np.transpose(eqn(x,np.transpose(w[j]),b[j])))
	op=np.transpose(eqn(xtest,np.transpose(w[j]),b[j]))
	op_wasted=np.transpose(eqn(xtest,np.transpose(w2[j]),b[j]))
	output_test.append(op)
	print('RMSE for neuron ',j,': ',np.sqrt(metrics.mean_squared_error(ytest,np.transpose(op))))
	print('RMSE for neuron ',j,': ',np.sqrt(metrics.mean_squared_error(y_waste_test,np.transpose(op_wasted))))
	print('\n')
	#output_crossvalidation.append(np.transpose(eqn(xcross,np.transpose(w[j]),b[j])))

print('enter no of children, adults and elders:')
ip=[]
for i in range(3):
	ip.append(int(input()))

ip=np.array(ip).reshape(3,1)

output=np.dot(w,ip)
wasted=np.dot(w2,ip)

s=np.sum(output)
s2=np.sum(wasted)

print('quantity of rice predicted(kgs): ',round(s/no_neurons,0))
print('quantity of rice expected to be wasted: ',round(s2/no_neurons,0))
	


