
# coding: utf-8

# In[38]:


import tensorflow as tf


# In[39]:


class SimpleRNN:
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        
        
        self.inputs=tf.placeholder(shape=[None,None,self.input_dim],dtype=tf.float32,name='inputs')
        
        #weights from input layer to hidden layer
        self.Wxh=tf.Variable(tf.random_normal(shape=[self.input_dim,self.hidden_dim],dtype=tf.float32))

        # weights from hidden layer to hidden layer 
        self.Whh=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.hidden_dim],dtype=tf.float32))

        # Bias for hidden layer

        self.bias=tf.Variable(tf.random_normal(shape=[self.hidden_dim],dtype=tf.float32))
        
        # weights from hidden layer to output layer
        self.Who=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.output_dim],dtype=tf.float32))

        # Bias for output layer
        self.bo=tf.Variable(tf.zeros(shape=[self.output_dim]))

        # Inputs are provided in the shape [batch_size,sequence_length,embedding_dim]
        # Converting it to [sequence_length,batch_size,embedding_dim]
        
        self.processed_inputs=tf.transpose(self.inputs,perm=[1,0,2])
        
        # Initializing an initial hidden state of shape [batch_size,hidden_dim]
        self.initial_hidden_state=tf.matmul(self.inputs[:,0,:],tf.zeros(shape=(self.input_dim,self.hidden_dim)))
        
    def step(self,hidden_state,x):
        

        '''

        Step function for RNN

        h[t]=tanh(x[t]*Wxh+Whh*h[t-1]+b)

        '''

        current_hidden_state=tf.matmul(x,self.Wxh)+tf.matmul(hidden_state,self.Whh)+self.bias
        current_hidden_state=tf.tanh(current_hidden_state)
        
        return current_hidden_state
    
    
    def get_all_states(self):

        #Unroll the RNN through time to get all hidden states.
        
        all_states=tf.scan(fn=self.step,elems=self.processed_inputs,initializer=self.initial_hidden_state)
        return all_states
    
    def get_output(self,hidden_state):

        # return output given hidden state
        
        out=tf.nn.relu(tf.matmul(hidden_state,self.Who)+self.bo)
        
        return out
    
    def get_all_outputs(self):

        # Unroll the RNN to get all outputs
        
        states=self.get_all_states()
        all_outputs=tf.map_fn(fn=self.get_output,elems=states)
        
        return all_outputs
    
    
    def batch_generator(self,X_train,Y_train,batch_size):

        # helper function to generate batches of data
        
        n_steps=int(len(X_train)/batch_size)
        
        for i in range(n_steps):
            yield X_train[i*batch_size:(i+1)*batch_size],Y_train[i*batch_size:(i+1)*batch_size]
        
        
    
    def train_on_sequences(self,X_train,Y_train,epochs,batch_size):
        
        out=self.get_all_outputs()[-1]
        y_ph=tf.placeholder(shape=[None,self.output_dim],dtype=tf.float32)
        loss=tf.losses.mean_squared_error(labels=y_ph,predictions=out)
        train=tf.train.AdamOptimizer().minimize(loss)
        
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        n_steps=int(len(X_train)/batch_size)
        
        for i in range(epochs):
            generator=self.batch_generator(X_train,Y_train,batch_size)
            for i in range(n_steps):
                x,y=next(generator)
                self.sess.run(train,feed_dict={self.inputs:x,y_ph:y})
            print('Loss on epoch {} = {}'.format(i,self.sess.run(loss,feed_dict={self.inputs:X_train,y_ph:Y_train})))
            
    def predict(self,x):
        
        out=self.get_all_outputs()[-1]
        outputs=self.sess.run(out,feed_dict={self.inputs:x})
        
        return outputs
        
        
        
        
        
        
                                            
                                        
                


# In[40]:


rnn=RNN(input_dim=1,hidden_dim=20,output_dim=1)


# In[33]:


import numpy as np


# In[34]:


a=np.arange(100)
b=np.array([a[i:i+5] for i in range(95)])
c=np.array([a[i+5] for i in range(95)])


# In[35]:


b=b.reshape(95,5,1)


# In[36]:


c=c.reshape(95,1)


# In[41]:


rnn.train_on_sequences(X_train=b,Y_train=c,epochs=2000,batch_size=10)


# In[26]:


rnn.predict(b[:10])


# In[25]:

