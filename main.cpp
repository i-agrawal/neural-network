#include "NeuralNetwork.h"

int main(int argc, char ** argv){
  vector<unsigned> setup;
  setup.push_back(2);
  setup.push_back(3);
  setup.push_back(3);
  setup.push_back(1);
  NeuralNetwork net(setup);
  vector<double> input_vals, target_vals, result_vals;

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

  for(unsigned i = 0; i < 10000; ++i){
    input_vals.clear();
    input_vals.push_back(rand() % 2);
    input_vals.push_back(rand() % 2);
    net.feed_forward(input_vals);
    net.get_results(result_vals);
    cout << "Average Error For Pass " << i+1 << ": " << net.get_error() << endl;
    target_vals.clear();
    double x = 0;
    if(input_vals[0] == input_vals[1])
      x = 0;
    else
      x = 1;

    target_vals.push_back(x);
    net.back_prop(target_vals);
  }

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
      cout << setprecision(1) << fixed << result_vals[i] << ", ";
    }
    cout << result_vals.back() << endl;
  }

  return 0;
}
