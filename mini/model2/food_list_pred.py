import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import json

data=pd.read_csv("/home/karthik/Desktop/mini/model2/train.csv",header=None,skiprows=1)

df=DataFrame(data)

x=df.iloc[:, 0:3]

children=pd.DataFrame(df.iloc[:, 0:1])
adults=pd.DataFrame(df.iloc[:, 1:2])
elderly=pd.DataFrame(df.iloc[:, 2:3])

gulabjamun=df.iloc[:, 3:4]
friedrice=df.iloc[:, 4:5]
whiterice=df.iloc[:, 5:6]
biriyani=df.iloc[:, 6:7]
chapathi=df.iloc[:, 7:8]
vegcurry=df.iloc[:, 8:9]
channamasala=df.iloc[:, 9:10]
pickle=df.iloc[:, 10:11]
chips=df.iloc[:, 11:12]
sweets=df.iloc[:, 12:13]
pakoda=df.iloc[:, 13:14]
icecream=df.iloc[:, 14:15]
noodles=df.iloc[:, 15:16]	
savouries=df.iloc[:, 16:17]

y=[gulabjamun,friedrice,whiterice,biriyani,chapathi,vegcurry,channamasala,pickle,chips,sweets,pakoda,icecream,noodles,
savouries]



output_amt=[]


#hyperparameters
no_neurons=14
m=x.shape[1]                                                  
no_examples=x.shape[0]
no_iter=500
alpha=0.00001


#x=np.transpose(x)
#weights is a matrix of 13X3  13 neurons ad 3 features

w=np.random.rand(m*no_neurons).reshape(no_neurons,m)  

b=np.ones((no_neurons,1))                                     
#w_wasted=np.random.rand(m*no_neurons).reshape(no_neurons,m)        


def sigmoid(z):
	s=1/(1+np.exp(-z))
	return s

def eqn(x,theta,bias):
	ans=np.dot(theta,x.T)
	#print(ans)
	return ans.reshape(x.shape[0],1)

def cost_func(m,y,pred):
	cost = -1/m*(np.dot(y,np.log(pred).T)+np.dot(1-y,np.log(1-pred).T))

def differential(m,y,pred,x):
	dw=np.dot(np.transpose(pred-y),x)/m
	#db=1/m*(np.sum(pred-y))
	#print(ans)
	return dw


def gradient_descent(m,x,y,theta,no_iter,alpha,bias):
	ind=[]
	cost=[]
	for i in range(no_iter):
		pred=sigmoid(eqn(x,theta,bias))
		dw=differential(m,y,pred,x)
		theta=theta-(alpha*dw)
		#bias=bias-(alpha*db)
	return theta.reshape(1,x.shape[1])


for i in range(no_neurons):
	w[i]=gradient_descent(no_examples,x,y[i],w[i],no_iter,alpha,b[i])


inp=[]
for i in range(3):
	inp.append(int(input('enter no of people')))

ratio=[]
ratio.append(inp[0]/inp[0])
ratio.append(inp[1]/inp[0])
ratio.append(inp[2]/inp[0])
ratio=np.array(ratio)
ratio.reshape(3,1)
ans=[]
for i in range(0,no_neurons):
	a=sigmoid(np.dot(w[i],ratio))
	if a<0.95:
		ans.append(0)
		i=i+1
	else:
		ans.append(1)
		i=i+1
	
#print(ans)  

#print(output_amt)
#ans=eqn()
calory={'gulab':225,'friedRice':319,'whiteRice':281,'biriyani':290,'chapathi':171,'vegCurry':169,
'channaMasala':130,'pickle':10,'chips':194,'sweet':304,'pakoda':122,'iceCream':173,'noodles':138,'savouries':245}

item=list(calory.keys())
menu=[]
#print(item)
cal=0
print('menu recommended: \n')
for i in range(len(ans)):
	if ans[i]==1:
		print(item[i])
		menu.append(item[i])
		cal=cal+calory[item[i]]

print('calorie count for menu: ',cal)

#no of calories for children,adults and elders
calorie_child=2028
calorie_adult=2645
calorie_elderly=2000

no_servings=int(((cal/calorie_child)*inp[0])+((cal/calorie_adult)*inp[1])+((cal/calorie_elderly)*inp[2]))

print('no of servings= ',no_servings)

with open('/home/karthik/Desktop/mini/model2/rawMaterials.json') as f:
  data = json.load(f)

for item in menu:
	for materials in data[item]:
 		data[item][materials]=data[item][materials]*no_servings

material={}
for item in menu:
	material[item]=data[item]

print('see materials to be bought in output.json file generated')
with open('output.json', 'w') as outfile:
    json.dump(material, outfile,indent=4)
