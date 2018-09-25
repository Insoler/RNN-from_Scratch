from SimpleRNN import SimpleRNN

class LSTM(SimpleRNN):
	def __init__(self,input_dim,hidden_dim,output_dim):


		super().__init__(input_dim, hidden_dim, output_dim)
		
		self.initial_cell_state=tf.matmul(self.inputs[:,0,:], tf.zeros(shape=[self.input_dim,self.hidden_dim]))
		self.initial_hidden_state=tf.matmul(self.inputs[:,0,:], tf.zeros(shape=[self.input_dim,self.hidden_dim]))

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


