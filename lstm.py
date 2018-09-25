import tensorflow as tf 


class LSTM:
	def __init__(self,hidden_dim,input_dim,output_dim):
		self.hidden_dim=hidden_dim
		self.input_dim=input_dim
		self.output_dim=output_dim

		self.inputs=tf.placeholder(dtype=tf.float32,shape=[None,None,self.input_dim])

		self.initial_cell_state=tf.matmul(self.inputs[:,0,:], tf.zeros(shape=[self.input_dim,self.hidden_dim]))
		self.initial_hidden_state=tf.matmul(self.inputs[:,0,:], tf.zeros(shape=[self.input_dim,self.hidden_dim]))

		self.processed_inputs=tf.transpose(self.inputs, perm=[1,0,2])

		self.Wfo=tf.Variable(tf.random_normal(shape=[self.input_dim,self.hidden_dim],dtype=tf.float32))
		self.Wfh=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.hidden_dim],dtype=tf.float32))
		self.bf=tf.Variable(tf.random_normal(shape=[self.hidden_dim],dtype=tf.float32))

		self.Wio=tf.Variable(tf.random_normal(shape=[self.input_dim,self.hidden_dim],dtype=tf.float32))
		self.Wih=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.hidden_dim],dtype=tf.float32))

		self.bi=tf.Variable(tf.random_normal(shape=[self.hidden_dim],dtype=tf.float32))


		self.Wco=tf.Variable(tf.random_normal(shape=[self.input_dim,self.hidden_dim],dtype=tf.float32))
		self.Wch=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.hidden_dim],dtype=tf.float32))

		self.bc=tf.Variable(tf.random_normal(shape=[self.hidden_dim],dtype=tf.float32))

		self.Woo=tf.Variable(tf.random_normal(shape=[self.input_dim,self.hidden_dim],dtype=tf.float32))
		self.Woh=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.hidden_dim],dtype=tf.float32))

		self.bo=tf.Variable(tf.random_normal(shape=[self.hidden_dim],dtype=tf.float32))




		self.Wout=tf.Variable(tf.random_normal(shape=[self.hidden_dim,self.output_dim],dtype=tf.float32))
		self.bout=tf.Variable(tf.random_normal(shape=[self.output_dim],dtype=tf.float32))


	def step(self,states,x):

		prev_cell_state=states[0]
		prev_hidden_state=states[1]
		
		#forget_gate

		
		ft=tf.matmul(x,self.Wfo)+tf.matmul(prev_hidden_state, self.Wfh)+self.bf
		ft=tf.nn.sigmoid(ft)

		new_cell_state=tf.multiply(prev_cell_state,ft)

		#input_gate


		it=tf.matmul(x, self.Wio)+tf.matmul(prev_hidden_state, self.Wih)+self.bi
		it=tf.nn.sigmoid(it)

		#update_gate


		C_t=tf.matmul(x, self.Wco)+tf.matmul(prev_hidden_state, self.Wch)+self.bc
		C_t=tf.tanh(C_t)

		new_cell_state=new_cell_state+tf.multiply(it, C_t)

		#output_gate


		ot=tf.matmul(x, self.Woo)+tf.matmul(prev_hidden_state, self.Woh)+self.bo
		ot=tf.nn.sigmoid(ot)

		new_hidden_state=tf.multiply(ot, tf.tanh(new_cell_state))

		return [new_cell_state,new_hidden_state]


	def get_all_states(self):

		states=tf.scan(fn=self.step, elems=self.processed_inputs,initializer=[self.initial_cell_state,self.initial_hidden_state])

		return states

	def get_output(self,hidden_state):

		output=tf.matmul(hidden_state, self.Wout)+self.bout

		output=tf.nn.relu(output)

		return output


	def get_all_outputs(self):

		states=self.get_all_states()
		hidden_states=states[1]

		outputs=tf.map_fn(fn=self.get_output, elems=hidden_states)

		return outputs

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














