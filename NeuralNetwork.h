#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace std;

//class to store training data WIP
class TrainingData{
public:
  TrainingData(const string &);
  bool eof(void);
  void get_setup(vector<unsigned> &);
  void get_next_inputs(vector<double> &);
  void get_target_outputs(vector<double> &);
private:
  ifstream m_data;
};

//struct to store neuron data
struct Neuron{
  Neuron(unsigned);

  vector<double> weights;
  vector<double> d_weights;
  double         m_value, m_gradient;
  static double rand_doub();
};

//stores multiple neurons
typedef vector<Neuron> layer;

//the neural network
class NeuralNetwork{
public:
  NeuralNetwork(const vector<unsigned> &);
  NeuralNetwork(const string &);
  void   feed_forward(const vector<double> &);
  void   back_prop(const vector<double> &);
  void   get_results(vector<double> &) const;
  void   save_net(const string &) const;
  double get_error(void) const;

private:
  static double act_function(double);
  static double act_function_prime(double);

  static double eta, alpha, error_smoothing;
  double        m_error, m_ave_error;
  vector<layer> m_network;
};
