#include "NeuralNetwork.h"

int main(int argc, char ** argv){
  // makes a nnet of 2 input, 2 hidden layers with 3 neurons each, and 1 output
  vector<unsigned> setup;
  setup.push_back(2);
  setup.push_back(3);
  setup.push_back(3);
  setup.push_back(1);
  NeuralNetwork net(setup);
  vector<double> input_vals, target_vals, result_vals;

// Will be used once training data is complete
  // TrainingData tdata(argv[0]);
  // vector<unsigned> setup;
  // tdata.get_setup(setup);
  //
  // NueralNetwork net(setup);
  //
  // vector<double> input_vals, target_vals, result_vals;
  // int npass = 0;
  //
  // while(!tdata.eof()){
  //   ++npass;
  //   tdata.get_next_inputs(input_vals);
  //   net.feed_forward(input_vals);
  //   net.get_results(result_vals);
  //   tdata.get_target_outputs(target_vals);
  //   net.back_prop(target_vals);
  //   cout << "Average Error For Pass " << npass << ": " << net.get_error() << endl;
  // }

  //basic traing cycle for xor over 100000 times
  for(unsigned i = 0; i < 100000; ++i){
    //get input
    input_vals.clear();
    input_vals.push_back(rand() % 2);
    input_vals.push_back(rand() % 2);

    //calculate output
    net.feed_forward(input_vals);

    //get the error
    cout << "Average Error For Pass " << i+1 << ": " << net.get_error() << endl;

    //get actual value
    target_vals.clear();

    //******XOR******
    double x = 0;
    if(input_vals[0] == input_vals[1])
      x = 0;
    else
      x = 1;
    //******XOR******

    target_vals.push_back(x);

    //adapt to new values
    net.back_prop(target_vals);
  }

  //check to see how good the neural net actually is
  //put in 2 numbers, 1's and 0's,
  /*
    0  0  |  0
    0  1  |  1
    1  0  |  1
    1  1  |  0
  */
  while(true){
    cout << "Try Inputting: ";
    input_vals.clear();
    for(unsigned i = 0; i < setup[0]; ++i){
      double input;
      cin >> input;
      input_vals.push_back(input);
    }

    net.feed_forward(input_vals);
    net.get_results(result_vals);
    cout << "Your Results: ";
    for(unsigned i = 0; i < result_vals.size() - 1; ++i){
      cout << result_vals[i] << ", ";
    }
    cout << result_vals.back() << endl;
  }

  return 0;
}
