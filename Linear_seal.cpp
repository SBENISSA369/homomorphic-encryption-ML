#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "seal.h"
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <algorithm>
using namespace std;
using namespace seal;

void read_csv(string csv_file_name,
	      vector < vector<float> > & csv_data){
  ifstream csv(csv_file_name);
  string line;
  string token;
  vector <float> temp;
  int k = 0;
  if (csv.is_open()) {
    while(csv.good()){
      getline(csv, line);
      istringstream buffer (line);
      while( getline( buffer, token, ',' ) ){
	temp.push_back(strtof((token).c_str(),0));
      }
      csv_data.push_back(temp);
      temp.clear();
    }
  }
}
float max_value(vector <float>& temp ){
  int k=0;
  float max = 0;
  while(k<temp.size()){
    if(temp[k]>max)
      max = temp[k];
    k=k+1;
  }
  return max;
}
void scale(vector < vector<float>> & csv_data,
	   vector < vector<float>> & data_norm){
  vector<float> maxi;
  vector<float> test;
  vector<float> temp;
  float max = 0;
  for(int j=0; j<csv_data[j].size(); ++j){
    for(int i=0; i<csv_data.size(); ++i){
      temp.push_back(csv_data[i][j]);
    }
    float max_v=max_value(temp);
    maxi.push_back(max_v);
    temp.clear();
  }
  for(int i=0; i<csv_data.size(); ++i){
    for(int j=0; j<csv_data[i].size(); ++j)
      test.push_back(csv_data[i][j]/maxi[j]);
    data_norm.push_back(test);
    test.clear();
  }
}
void split(vector < vector<float>> & data_norm,
	   vector < vector<float>> & training,
	   vector < vector<float>> & testing){
  vector<int> indice;
  float t = data_norm.size() * 0.2;
  srand ( unsigned ( time(0) ) );
  for (int i=0; i<data_norm.size(); ++i) indice.push_back(i);
  random_shuffle(indice.begin(), indice.end());
  for(int j =0; j<indice.size();++j){
    if(j < t)  testing.push_back(data_norm[j]);
    else     training.push_back(data_norm[j]);
  }
}

void encrypt_data(vector<vector<float>> &csv_data,
		  vector<vector<Ciphertext>> &data_enc, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  vector<Ciphertext> encrypted_temp;
   for (int x = 0; x < csv_data.size(); ++x){
    for (int y = 0; y < csv_data[x].size(); ++y){
      Plaintext encoded_number = encoder.encode(csv_data[x][y]);
      encrypted_temp.emplace_back(encryptor.encrypt(encoded_number));
    }
    data_enc.push_back(encrypted_temp);
    encrypted_temp.clear();
  }
}
void func_vect_enc(vector<float>& X, vector<float>& model, float& vect_clair,  vector<Ciphertext>& X_enc, vector<Ciphertext>& model_enc, float& vect_enc, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  vector<Ciphertext> vect_somme;
  vect_enc = 0;
  vect_clair = 0;
  vect_clair = model[0];
  vect_somme.push_back(model_enc[0]);
  for(int i=0 ; i< X_enc.size()-1 ; ++i){
       model_enc[i+1] = evaluator.relinearize(model_enc[i+1]);
       Ciphertext mult = evaluator.multiply(X_enc[i], model_enc[i+1]);
       mult = evaluator.relinearize(mult);
       vect_clair += X[i] * model[i+1]; 
       vect_somme.push_back(mult);
  }
  Ciphertext somme = evaluator.add_many(vect_somme);
  vect_enc = encoder.decode(decryptor.decrypt(somme));
  vect_somme.clear();

}
void SGD(vector<float>& sample,
	 vector<float>& model,
	 float& lr,
	 vector<Ciphertext>& sample_enc,
	 vector<Ciphertext>& model_enc,Plaintext& encoded_lr, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  float vect_enc = 0;
  float vect_clair = 0;
  func_vect_enc(sample, model , vect_clair, sample_enc, model_enc, vect_enc, encryptor, decryptor, evaluator, encoder);
  Ciphertext vect_class = evaluator.negate(evaluator.sub_plain( sample_enc[sample_enc.size()-1], encoder.encode(vect_enc)));
  float vect_class_clair =  vect_clair - sample[sample.size()-1];
  Ciphertext vect_class_lr = evaluator.multiply_plain(vect_class, encoded_lr);
  float vect_class_lr_clair = vect_class_clair * lr;
  model_enc[0] = evaluator.sub(model_enc[0], vect_class_lr);
  model[0] = model[0] - vect_class_lr_clair;
  for( int i = 1; i < model_enc.size(); i++){
    Ciphertext vect_class_mult = evaluator.multiply(vect_class_lr, sample_enc[i]);
	   float vect_class_mult_clair = vect_class_lr_clair * sample[i];
           model_enc[i] = evaluator.sub(model_enc[i], vect_class_mult );
	   model[i] = model[i] - vect_class_mult_clair;
	  }
}
void erreur(vector<vector<float>>& testing, vector<float>& model, float& loss_clair, vector<vector<Ciphertext>>& testing_enc, vector<Ciphertext>& model_enc, float& loss_enc, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  vector<Ciphertext> Erreur;
  loss_enc = 0;
  float Erreur_ = 0;
   float vect_enc =  0;
  float vect_clair = 0;
  for(int i=0; i< testing.size(); ++i){
    func_vect_enc(testing[i], model, vect_clair, testing_enc[i], model_enc, vect_enc, encryptor, decryptor, evaluator, encoder);
    Ciphertext temp_1 = evaluator.negate(evaluator.sub_plain(testing_enc[i][testing_enc[i].size()-1], encoder.encode(vect_enc)));
    float temp_1_clair = testing[i][testing[i].size()-1] - vect_clair;
    Erreur_ +=  temp_1_clair * temp_1_clair;
    Erreur.push_back(evaluator.multiply(temp_1, temp_1));
  }
    Ciphertext temp_2 = evaluator.add_many(Erreur);
    Erreur.clear();
    loss_clair = (Erreur_/testing.size());
    loss_enc = (encoder.decode(decryptor.decrypt(temp_2))/testing_enc.size());
}
void train(vector<vector<float>>& testing,vector<vector<float>>& training,
	   vector<float>& model,vector<vector<Ciphertext>>& testing_enc,
	   vector<vector<Ciphertext>>& training_enc,                                                                                                           
	   vector<Ciphertext>& model_enc, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  int iter = 0;
  float loss_enc = 0;
  float loss_clair = 0;
  float lr = 0.01;
  Plaintext encoded_lr = encoder.encode(0.01);
  string const nomFichier("loss.csv");
  ofstream monFlux(nomFichier.c_str());
  while(iter<1){
    for(int i = 0; i< training.size(); ++i){
      cout<<"-------------------"<<iter*i+i<<"-----------------------------"<<endl;
                SGD(training[i], model, lr,training_enc[i], model_enc, encoded_lr, encryptor, decryptor, evaluator, encoder);
		//for(int k=0; k<model_enc.size(); ++k)  model_enc[i] = encryptor.encrypt(encoder.encode(encoder.decode(decryptor.decrypt(model_enc[k]))));
		for(int j=0; j < model_enc.size(); ++j)
		cout<<"model_enc ["<<j<<"] = "<<(encoder.decode(decryptor.decrypt(model_enc[j])))<<"||||||||||||"<<"model["<<j<<"] = "<<model[j]<<endl;
	        erreur(testing, model, loss_clair, testing_enc, model_enc, loss_enc, encryptor, decryptor, evaluator, encoder);
		monFlux << iter << " , " << loss_enc << " , " <<loss_clair<< endl;
	        cout<<"loss_enc = "<<loss_enc<<endl;
		cout<<"loss_clair = "<<loss_clair<<endl;
							         }
     iter=iter+1;
  }
 }                                                                                                                                                                
void model_encryption(vector<float>& model,
		      vector<Ciphertext>& model_enc, Encryptor encryptor, Decryptor decryptor, Evaluator evaluator,FractionalEncoder encoder){
  for (int y = 0; y < model.size(); ++y){
    Plaintext encoded_number = encoder.encode(model[y]);
    model_enc.emplace_back(encryptor.encrypt(encoded_number));
  }
}

int main (){
  EncryptionParameters parms;
  parms.set_poly_modulus("1x^4096 + 1");
  parms.set_coeff_modulus(ChooserEvaluator::default_parameter_options().at(4096));
  parms.set_plain_modulus(40961);
  parms.set_decomposition_bit_count(16);
  parms.validate();
  KeyGenerator generator(parms);
  generator.generate(1);
  Ciphertext public_key = generator.public_key();
  Plaintext secret_key = generator.secret_key();
  EvaluationKeys evaluation_keys = generator.evaluation_keys();
  FractionalEncoder encoder(parms.plain_modulus(), parms.poly_modulus(), 64, 32, 2);
  Encryptor encryptor(parms, public_key);
  Decryptor decryptor(parms, secret_key);
  Evaluator evaluator(parms, evaluation_keys);
  vector<vector<float> > csv_data;
  vector<vector<float>> data_norm;
  vector<float> model{0.2,0.3,0.4,0.7,0.9,0.1,0.7,0.2,0.2};
  vector<Ciphertext> model_enc;
  vector< vector<Ciphertext> > data_enc;
  vector< vector<float>> training;
  vector< vector<float>> testing;
  vector< vector<Ciphertext>> training_enc;
  vector< vector<Ciphertext>> testing_enc;
  string csv_file_name = "pima.csv";
  read_csv(csv_file_name,csv_data);
  scale(csv_data, data_norm);
  split(data_norm, training, testing);
  encrypt_data(training, training_enc, encryptor, decryptor, evaluator, encoder);
  encrypt_data(testing, testing_enc, encryptor, decryptor, evaluator, encoder);
  model_encryption(model, model_enc,encryptor, decryptor, evaluator, encoder);
  train (testing, training, model, testing_enc, training_enc, model_enc,  encryptor, decryptor, evaluator, encoder);
  return 0;
}
