from matplotlib import pyplot as plt
from numpy import *

def Trans_datasets(filename):
      file=open(filename)
      X=[]
      y=[]
      nums=[]
      for line in file.readlines():
            str_num=line.strip().split()
            nums=[float(s) for s in str_num]
            y.append(int(nums[-1]))
            X.append(nums[:-1])
      return array(X),array(y)

def avgC(X,k,cluster):
      res=[]
      for i in range(k):
            cluster[i]=list(cluster[i])
            size=len(cluster[i])  ##  聚类中数据数
            sum_dist=0
            for m in range(size):
                  for n in range(m+1,size):
                        sum_dist+=Euclid_dist(X[m],X[n])
            res.append(sum_dist/size/(size-1))
      return res

def DBindex(X,k,cluster,mean_vectors):
      DBI=0
      reference=zeros((k,k),dtype='float64')
      Cavg=avgC(X,k,cluster)
      for i in range(k):
            for j in range(i+1,k):
                  reference[i][j]=(Cavg[i]+Cavg[j])/Euclid_dist(mean_vectors[i],mean_vectors[j])
      reference+=reference.T
      for i in range(k):
            DBI+=reference[i].mean()
      return DBI/k

def Euclid_dist(x,y):
      return sqrt(sum((x-y)**2))

def nearest_vector(x,vectors):
      dist=inf
      nearest=inf
      for i in range(3):
            temp_d=Euclid_dist(x,vectors[i])
            if temp_d<dist:
                  dist=temp_d
                  nearest=i
      return nearest

def E_step(X,mean_vectors,cluster):
      n=len(mean_vectors)
      for i in range(len(X)):
            min_dist=inf
            des=inf
            for j in range(n):
                  d=Euclid_dist(X[i],mean_vectors[j])
                  if d<min_dist:
                        min_dist=d
                        des=j  ##  归类地
            cluster[des].append(i)

def M_step(X,mean_vectors,cluster):
      for i in range(len(cluster)):
            mean_vectors[i]=mean(X[cluster[i]],axis=0)
     
def clustering(X,mean_vectors):
      n=len(mean_vectors)
      rounds=100 ##  迭代轮次
      while True:
            cluster=[]
            for i in range(n):  ##   k个集合
                  cluster.append([])
            
            E_step(X,mean_vectors,cluster)
            
            M_step(X,mean_vectors,cluster)
            rounds-=1
            if rounds==0:
                  break
      return cluster,mean_vectors
      
def Kmeans(X,k):
      m,n=shape(X)
      init_vectors_index=random.choice(list(range(m)),size=k,replace=False)
      init_vectors=[]   
      for i in init_vectors_index:
            init_vectors.append(X[i])
      cluster,final_mean=clustering(X,init_vectors)
      return cluster,final_mean

def visualize(cluster):
      y=ones(210,dtype='int8')
      for i in range(1,3):
            y[list(cluster[i])]=i+1
      return y

def LVQ(X,y,eta=1e-3):
      init_vector_index=[0,70,140]
      init_vectors=X[init_vector_index]
      rounds=1500 ##  迭代轮次
      cluster=[set([]),set([]),set([])]
      while True:
            index=random.randint(210)
            nearest=nearest_vector(X[index],init_vectors)
            if y[index]==nearest+1:  ##类别相同
                  init_vectors[nearest]+=eta*(X[index]-init_vectors[nearest])
            else:
                  init_vectors[nearest]-=eta*(X[index]-init_vectors[nearest])
            rounds-=1
            if rounds==0:
                  break
            
      ##划分势力范围
      for i in range(210):
            nearest=nearest_vector(X[i],init_vectors)
            cluster[nearest].add(i)
      return cluster,init_vectors

def Gaussian(x,means,covariance):
      return exp(-0.5*(x-means).dot(linalg.inv(covariance)).dot(x-means))/sqrt(linalg.det(covariance))

def GMM(X,init_vectors,cluster):  ##  在K-means基础上继续迭代
      alpha=[]
      covariance=[]
      means=init_vectors
      post=zeros((len(X),3))
      rounds=200
      for i in range(3):
            alpha.append(len(cluster[i])/210)
            covariance.append(cov(X[list(cluster[i])],rowvar=False))
            
      while True:
            ##计算后验概率
            for j in range(210):
                  post_sum=0
                  for i in range(3):
                        temp=alpha[i]*Gaussian(X[j],means[i],covariance[i])
                        post[j][i]=temp
                        post_sum+=temp
                  for i in range(3):
                        post[j][i]/=post_sum 
            ##更新参数
            for i in range(3):
                  post_sum=sum(post[:,i])
                  means[i]=post[:,i].dot(X)/post_sum
                  covariance[i]=(post[:,i]*(X-means[i]).T).dot((X-means[i]))/post_sum
                  alpha[i]=post_sum/210
            rounds-=1
            if rounds==0:
                  break
      return argmax(post,axis=1)+1, means

def kmeans_gmm_lvq():
      ##标准化数据
      filename=u'seeds.txt'
      X,y=Trans_datasets(filename)
      X=(X-X.mean(axis=0))/X.std(axis=0)

      ##PCA至二维便于可视化
      values,vectors=linalg.eig(X.T.dot(X))
      W=vectors[:,:2]  ##投影矩阵
      newX=X.dot(W)

      ##原图
      plt.figure(1)
      plt.scatter(newX[:,0],newX[:,1],s=10,c=y,cmap=plt.cm.RdYlBu_r)

      ##K-means效果图
      cluster,final_mean=Kmeans(X)
      newy=visualize(cluster)
      new_mean=array(final_mean).dot(W)
      plt.figure(2)
      plt.scatter(new_mean[:,0],new_mean[:,1],s=100,c='r')
      plt.scatter(newX[:,0],newX[:,1],s=10,c=newy,cmap=plt.cm.RdYlBu_r)
      print('K-means的DBI为:',DBindex(X,3,cluster,final_mean))

      ##GMM效果图
      plt.figure(3)
      y_GMM,mean_GMM=GMM(X,final_mean,cluster)
      mean_GMM=array(mean_GMM).dot(W)
      for i in range(3):
            cluster[i]=where(y_GMM==i+1)[0]
      plt.scatter(mean_GMM[:,0],mean_GMM[:,1],s=100,c='r')
      plt.scatter(newX[:,0],newX[:,1],s=10,c=y_GMM,cmap=plt.cm.RdYlBu_r)
      print('GMM的DBI为:',DBindex(X,3,cluster,mean_GMM))

      ##LVQ效果图
      plt.figure(4)
      cluster,prototype=LVQ(X,y)
      y_lvq=visualize(cluster)
      mean_lvq=array(prototype).dot(W)
      plt.scatter(mean_lvq[:,0],mean_lvq[:,1],s=100,c='r')
      plt.scatter(newX[:,0],newX[:,1],s=10,c=y_lvq,cmap=plt.cm.RdYlBu_r)
      print('LVQ的DBI为:',DBindex(X,3,cluster,mean_lvq))

      plt.show()

def image_compression():
      A=plt.imread('bird_small.png')
      plt.subplot(231)
      plt.imshow(A)  ##原始图像
      K=[3,6,10,16,30]
      for k in range(len(K)):
            A=plt.imread('bird_small.png')  ##非得在这重写一遍，否则一直出错，找了半天！
##            print(A[1])  ##不写上面一句，A就彻底变样！
            B=A.reshape((128*128,3))
            cluster,means=Kmeans(B,K[k])  ##cluster是列表，元素为列表;means也是列表，元素array
            for i in range(K[k]):
                  B[list(cluster[i])]=means[i]
            C=B.reshape(A.shape)
            plt.subplot(2,3,k+2)
            plt.title('k=%d'%K[k])
            plt.imshow(C)
      plt.savefig('Compression.png')
      plt.show()

if __name__=='__main__':
      image_compression()














