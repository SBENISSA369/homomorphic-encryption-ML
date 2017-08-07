#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "seal.h"
#include <time.h>
#include <sys/time.h>
#include <fstream>
using namespace std;
using namespace seal;
void linear_regression();
int main()
{
  linear_regression();
}
float calcul_Erreur(float theta_0, float theta_1, float theta_2, vector<float> & X_1, vector<float> & X_2 , vector<float> & Y, int N){
  cout<<"theta_0 = "<<theta_0<<endl;
  cout<<"theta_1 = "<<theta_1<<endl;
  cout<<"theta_2 = "<<theta_2<<endl;
  cout<<"------------------------------------------------------"<<endl;
  vector <float> somme;
  float somme_ = 0;
  EncryptionParameters parms;
  parms.set_poly_modulus("1x^2048 + 1");
  parms.set_coeff_modulus(ChooserEvaluator::default_parameter_options().at(2048));
  parms.set_plain_modulus(1 << 4);
  parms.set_decomposition_bit_count(16);
  parms.validate();
  KeyGenerator generator(parms);
  generator.generate(1);
  Ciphertext public_key = generator.public_key();
  Plaintext secret_key = generator.secret_key();
  EvaluationKeys evaluation_keys = generator.evaluation_keys();
  FractionalEncoder encoder(parms.plain_modulus(), parms.poly_modulus(), 64, 32, 3);
  Encryptor encryptor(parms, public_key);
  Decryptor decryptor(parms, secret_key);
  Evaluator evaluator(parms, evaluation_keys);
  Plaintext encodedTheta_0 = encoder.encode(theta_0);
  Plaintext encodedTheta_1 = encoder.encode(theta_1);
  Plaintext encodedTheta_2 = encoder.encode(theta_2);
  Ciphertext Erreur = encryptor.encrypt(encoder.encode(0));
  vector<Ciphertext> encrypted_X_1;
  vector<Ciphertext> encrypted_X_2;
  vector<Ciphertext> encrypted_Y;
  for (int i = 0; i < 4; ++i){
    Plaintext encoded_X_1 = encoder.encode(X_1[i]);
    Plaintext encoded_X_2 = encoder.encode(X_2[i]);
    Plaintext encoded_Y = encoder.encode(Y[i]);
    encrypted_X_1.emplace_back(encryptor.encrypt(encoded_X_1));
    encrypted_X_2.emplace_back(encryptor.encrypt(encoded_X_2));
    encrypted_Y.emplace_back(encryptor.encrypt(encoded_Y));
  }
  for (int i=0; i < N ; i++){
  Ciphertext mult1 = evaluator.multiply_plain(encrypted_X_1[i], encodedTheta_1);
  Ciphertext mult2 = evaluator.multiply_plain(encrypted_X_2[i], encodedTheta_2);
  Ciphertext mult1_mult2 = evaluator.add(mult1, mult2);
  Ciphertext add_mult1_mult2 = evaluator.add_plain(mult1_mult2, encodedTheta_0 );
  Ciphertext diff = evaluator.sub(encrypted_Y[i], add_mult1_mult2);
  Ciphertext square = evaluator.square(diff);
  Plaintext decoded_Erreur= decryptor.decrypt(square);
  float Erreur_ = encoder.decode(decoded_Erreur);
  somme.push_back(Erreur_);
  cout<<"Erreur_ = "<<Erreur_<<endl;
 }
  for (int i = 0; i != somme.size(); i++) somme_ += somme[i];
  somme.clear();
  cout<<"return ="<<(somme_/float(N))<<endl;
  return (somme_/float(N));
  
}

void linear_regression(){
  vector<float> X_1 {0,0,1,1};
  vector<float> X_2 {0,1,0,1};
  vector<float> Y {-1,1,1,1};
  int i;
  int N=4;
  float learningRate = 0.001;
  int iter = 0;
  float theta_0 = 1;
  float theta_1 = 1;
  float theta_2 = 1;
  float Erreur;
  EncryptionParameters parms;
  parms.set_poly_modulus("1x^2048 + 1");
  parms.set_coeff_modulus(ChooserEvaluator::default_parameter_options().at(2048));
  parms.set_plain_modulus(1 << 4);
  parms.set_decomposition_bit_count(16);
  parms.validate();
  cout << "Generating keys ..." << endl;
  KeyGenerator generator(parms);
  generator.generate(1);
  cout << "... key generation complete" << endl;
  Ciphertext public_key = generator.public_key();
  Plaintext secret_key = generator.secret_key();
  EvaluationKeys evaluation_keys = generator.evaluation_keys();
  FractionalEncoder encoder(parms.plain_modulus(), parms.poly_modulus(), 64, 32, 3);
  Encryptor encryptor(parms, public_key);
  Decryptor decryptor(parms, secret_key);
  Evaluator evaluator(parms, evaluation_keys);
  Plaintext encodedTheta_0 = encoder.encode(theta_0);
  Plaintext encodedTheta_1 = encoder.encode(theta_1);
  Plaintext encodedTheta_2 = encoder.encode(theta_2);
  Plaintext encodedLearningRate = encoder.encode(learningRate);
  Ciphertext  encryptedtheta_0 = encryptor.encrypt(encodedTheta_0) ;
  Ciphertext  encryptedtheta_1 = encryptor.encrypt(encodedTheta_1) ;
  Ciphertext  encryptedtheta_2 = encryptor.encrypt(encodedTheta_2) ;
  vector<Ciphertext> encrypted_X_1;
  vector<Ciphertext> encrypted_X_2;
  vector<Ciphertext> encrypted_Y;
  for (int i = 0; i < 4; ++i){
    Plaintext encoded_X_1 = encoder.encode(X_1[i]);
    Plaintext encoded_X_2 = encoder.encode(X_2[i]);
    Plaintext encoded_Y = encoder.encode(Y[i]);
    encrypted_X_1.emplace_back(encryptor.encrypt(encoded_X_1));
    encrypted_X_2.emplace_back(encryptor.encrypt(encoded_X_2));
    encrypted_Y.emplace_back(encryptor.encrypt(encoded_Y));
  }
    string const nomFichier("erreur_dataset_encrypter.csv");
    ofstream monFlux(nomFichier.c_str());
        while (iter < 8500){
	  cout<<"-------------------------------------------"<<endl;
	  cout <<"("<<iter+1<<")"<<"itération"<<endl;
    for(i = 0;i<4; i++){
      encodedTheta_0 = encoder.encode(theta_0);
      encodedTheta_1 = encoder.encode(theta_1);
      encodedTheta_2 = encoder.encode(theta_2);
      Ciphertext mult_theta1 = evaluator.multiply_plain(encrypted_X_1[i], encodedTheta_1);
      Ciphertext mult_theta2 = evaluator.multiply_plain(encrypted_X_2[i], encodedTheta_2);
      Ciphertext add_mult1_mult2 = evaluator.add(mult_theta1, mult_theta2);
      Ciphertext inter1 = evaluator.add_plain(add_mult1_mult2, encodedTheta_0);
      Ciphertext inter2 = evaluator.sub(inter1 , encrypted_Y[i]);
      Ciphertext inter3 = evaluator.multiply_plain(inter2, encodedLearningRate);
      Ciphertext inter4 = evaluator.sub_plain(inter3,  encodedTheta_0);
      encryptedtheta_0 = evaluator.negate(inter4);
      Plaintext encodedtheta_0 = decryptor.decrypt(encryptedtheta_0);
      theta_0 = encoder.decode(encodedtheta_0);
      Ciphertext inter5 = evaluator.multiply(inter3, encrypted_X_1[i]);
      Ciphertext inter6 = evaluator.sub_plain(inter5, encodedTheta_1);
      encryptedtheta_1 = evaluator.negate(inter6);
      encodedTheta_1 = decryptor.decrypt(encryptedtheta_1);
      theta_1 = encoder.decode(encodedTheta_1);
      Ciphertext inter7 = evaluator.multiply(inter3, encrypted_X_2[i]);
      Ciphertext inter8 = evaluator.sub_plain(inter7, encodedTheta_2);
      encryptedtheta_2 = evaluator.negate(inter8);
      encodedTheta_2 = decryptor.decrypt(encryptedtheta_2);
      theta_2 = encoder.decode(encodedTheta_2);
            }
    iter = iter + 1;
    Erreur = calcul_Erreur(theta_0, theta_1, theta_2, X_1, X_2, Y,N);
    cout <<"Erreur = "<<Erreur<<endl;
    monFlux << iter << " , " << Erreur << " ; " << endl;
    
  }
  cout<<"\n"<<endl;
  printf("theta_0 = %lf \n", theta_0);
  printf("theta_1 = %lf \n", theta_1);
  printf("theta_2 = %lf \n", theta_2);
  
}

