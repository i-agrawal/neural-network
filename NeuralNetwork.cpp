#include "NeuralNetwork.h"

TrainingData::TrainingData(const string &file){
  m_data.open(file.c_str());
}

void TrainingData::get_setup(vector<unsigned> &setup){
  //however I wanna make my file
}

void TrainingData::get_next_inputs(vector<double> &input_vals){
  //however I wanna make my file
}

void TrainingData::get_target_outputs(vector<double> &target_vals){
  //however I wanna make my file
}


Neuron::Neuron(unsigned outs){
  for(unsigned i = 0; i < outs; ++i){
    //the neuron has a random starting weight for each weight
    weights.push_back(rand_doub());
    //with no change in weight yet
    d_weights.push_back(0.0);
  }
}

//produces a randome double
double Neuron::rand_doub(){
  return rand() / double(RAND_MAX);
}

//the learning rate = eta, the momentum = alpha, the smoothing = over what period of errors
double NeuralNetwork::eta = 0.15;
double NeuralNetwork::alpha = 0.5;
double NeuralNetwork::error_smoothing = 100.0;

//the activation function as hyperbolic tangent
double NeuralNetwork::act_function(double x){
  return tanh(x);
}

//the derivative of the activation function rounded to 1 - x^2 for speed rather than 1 - (tanh(x))^2
double NeuralNetwork::act_function_prime(double x){
  return 1.0 - x * x;
}

//uses a vector with a number of neurons corresponding to each layer
NeuralNetwork::NeuralNetwork(const vector<unsigned> &setup){
  //for loop to create each layer
  for(unsigned i = 0; i < setup.size(); ++i){
    //add a layer to the vector holding our layers
    m_network.push_back(layer());
    //find the number of outputs in the next layer excpet for last layer
    unsigned n_outs = 0;
    if(i != setup.size() - 1)
      n_outs = setup[i + 1];
    else
      n_outs = 0;

    //add a Neuron with the amound of weights needed to output to the next layer
    //add an extra Neuron (<=) for the bias
    for(unsigned j = 0; j <= setup[i]; ++j)
      m_network[i].push_back(Neuron(n_outs));

    //the bias always has a value fo 1.0
    m_network[i].back().m_value = 1.0;
  }
}

//work in progress to initialize a neural network from file
NeuralNetwork::NeuralNetwork(const string &file){
  ifstream input(file.c_str());

  //however I wanna make my file
}

//uses a vector of input values to create output values
void NeuralNetwork::feed_forward(const vector<double> &input_vals){
  //for loop that sets the values for the input layer corresponding to the inputted values
  for(unsigned i = 0; i < m_network[0].size() - 1; ++i){
    m_network[0][i].m_value = input_vals[i];
  }

  //goes through each layer
  for(unsigned i = 1; i < m_network.size(); ++i){
    //goes through each neuron in the layer
    for(unsigned j = 0; j < m_network[i].size() - 1; ++j){
      //finds the value by adding up all the neurons in the previous layer's value * the specific weight
      double sum = 0.0;
      for(int k = 0; k < m_network[i - 1].size(); ++k){
        sum += m_network[i - 1][k].m_value * m_network[i - 1][k].weights[j];
      }
      //find the neurons value using the activation function
      m_network[i][j].m_value = act_function(sum);
    }
  }
}

void NeuralNetwork::back_prop(const vector<double> &target_vals){
  //we need to find the mean square error = mse
  m_error = 0.0;
  //the layer with the results calculated by the neural network
  layer &output_layer = m_network.back();

  //for each output
  for(unsigned i = 0; i < output_layer.size() - 1; ++i){
    //get the difference in the actual value and the calculated value
    double delta = target_vals[i] - output_layer[i].m_value;
    //sum the squares
    m_error += delta * delta;
  }
  // divide it by the size, - 1 because of bias
  m_error /= output_layer.size() - 1;
  // square root it for the mse
  m_error = sqrt(m_error);

  //finds the average error over the last 100 calculations
  m_ave_error = (m_ave_error * error_smoothing + m_error) / (error_smoothing + 1);

  //calculate the gradient
  for(unsigned i = 0; i < output_layer.size() - 1; ++i){
    //it is the difference between actual and calculated
    double delta = target_vals[i] - output_layer[i].m_value;
    //time the derivative of the activation function of the calculate value
    output_layer[i].m_gradient = delta * act_function_prime(output_layer[i].m_value);
  }

  //calculate the gradient starting from 2 to last layer
  for(unsigned i = m_network.size() - 2; i > 0; --i){
    //fo through every neuron
    for(unsigned j = 0; j < m_network[i].size(); ++j){
      //sum the gradients from the layer before it times the weight of the current neuron
      double sum = 0.0;
      for(unsigned k = 0; k < m_network[i + 1].size() - 1; ++k){
        sum += m_network[i][j].weights[k] * m_network[i + 1][k].m_gradient;
      }
      //multiply the sum by the derivative of the activation function and that it the gradient
      m_network[i][j].m_gradient = sum * act_function_prime(m_network[i][j].m_value);
    }
  }

  //go through every layer
  for(unsigned i = m_network.size() - 1; i > 0; --i){
    //go through every neuron
    for (unsigned j = 0; j < m_network[i].size() - 1; ++j) {
      //go through the layer before it
      for(unsigned k = 0; k < m_network[i - 1].size(); ++k){
        //find its old delta weight = change in weight
        double od_weight = m_network[i - 1][k].d_weights[j];
        //find the new change in weight
        double nd_weight = eta * m_network[i - 1][k].m_value * m_network[i][j].m_gradient + alpha * od_weight;
        //the new change in weight will become the delta weight
        m_network[i - 1][k].d_weights[j] = nd_weight;
        //add the change in weight to the weight
        m_network[i - 1][k].weights[j] += nd_weight;
      }
    }
  }
}

//stores the resulting values from feed-forward
void NeuralNetwork::get_results(vector<double> &result_vals) const{
  result_vals.clear();
  for(unsigned i = 0; i < m_network.back().size() - 1; ++i){
    result_vals.push_back(m_network.back()[i].m_value);
  }
}

//work in progress, going to try to save neural net data into a file
void NeuralNetwork::save_net(const string &file) const{
  ofstream output(file.c_str());
  if(output.is_open()){
    for(unsigned i = 0; i < m_network.size(); ++i)
      output << m_network[i].size() - 1 << " ";
    output << endl;


  }
  else{
    cout << "Failed to Save Neural Network" << endl;
  }
}

//simple function that returns the error
double NeuralNetwork::get_error() const{
  return m_error;
}




//space
