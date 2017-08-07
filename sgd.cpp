#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <cstdlib>
using namespace std;
using namespace Eigen;
float calcul_Erreur(float theta_0, float theta_1, float theta_2, vector<float> & X_1, vector<float> & X_2, vector<float> & Y_i, int N){
  cout<<"theta_0 = "<<theta_0<<endl;
  cout<<"theta_1 = "<<theta_1<<endl;
  cout<<"theta_2 = "<<theta_2<<endl;
  cout<<"-------------------------------------------------"<<endl;
  float Erreur = 0;
           int x,y,z;
           for (int i=0; i < N ; i++){
	     cout<<"Erreur_ = "<<(( Y_i[i] - (theta_0 + theta_1 * X_1[i] + theta_2 * X_2[i])) * ( Y_i[i]- (theta_0 + theta_1 * X_1[i] + theta_2 * X_2[i])))<<endl;
                      Erreur += (( Y_i[i] - (theta_0 + theta_1 * X_1[i] + theta_2 * X_2[i])) * ( Y_i[i]- (theta_0 + theta_1 * X_1[i] + theta_2 * X_2[i])));
	   }
	   cout<<"return = "<<(Erreur/float(N))<<endl;
	   return (Erreur/float(N));
	   
}
std::vector<float> stochastic_gradient_descent(vector<float> & X_1, vector<float> &  X_2, vector<float> &  Y, int N){
           vector <float> theta {1,1,1};
	   int iter = 0;
	   float Erreur;
	   float learningRate = 0.001;
	   int i = 0, k = 0;
	   int v1;
	   string const nomFichier("erreur_dataset_clair.csv");
	   ofstream monFlux(nomFichier.c_str());
	   while (iter < 8500){
	     cout<<"-------------------------------------------"<<endl;
	     cout <<"("<<iter+1<<")"<<"itération"<<endl;
	             for (i=0 ; i<4 ; i++){
		                     theta[0]=theta[0]-learningRate*(theta[0] + theta[1]*X_1[i] + theta[2]*X_2[i] - Y[i]);
				     theta[1]=theta[1] - learningRate*(theta[0] + theta[1]*X_1[i] + theta[2]*X_2[i] - Y[i])*X_1[i];
				     theta[2]=theta[2] - learningRate *(theta[0] + theta[1]*X_1[i] + theta[2]*X_2[i] - Y[i])*X_2[i];
                        
		     }
		     	   iter = iter + 1;
	    Erreur = calcul_Erreur(theta[0],theta[1],theta[2], X_1, X_2, Y,N);
	    cout <<"Erreur = "<<Erreur<<endl;
	   monFlux << iter << " , " << Erreur << " ; " << endl;
           }
	   return theta;
}
std::vector<float> iteration(float actual_theta_0, float actual_theta_1, float actual_theta_2,float learningRate, vector<float> &  X_1, vector<float> &  X_2,vector<float> &  Y_i, int N){
           vector <float> theta(3);
	   float  theta_0_gradient = 0;
	   float  theta_1_gradient = 0;
	   float  theta_2_gradient = 0;
	   float  m_gradient = 0;
	   int x, y, z;
	   int i = 0;
	   int iter = 0;
	   for (i=0;i < N; i++){
                 x = X_1[i];
		 y = X_2[i];
		 z = Y_i[i];
		 theta_0_gradient += -(2/(float)N) * (z - (actual_theta_0 + (actual_theta_1 * x) + (actual_theta_2 * y) ));
		 theta_1_gradient += -(2/(float)N) * x * (z - (actual_theta_0 + (actual_theta_1 * x) + (actual_theta_2 * y) ));
		 theta_2_gradient += -(2/(float)N) * y * (z - (actual_theta_0 + actual_theta_1 * x + actual_theta_2 * y ));        
	   }
	   theta[0] = actual_theta_0 - (learningRate * theta_0_gradient);
	   theta[1] = actual_theta_1 - (learningRate * theta_1_gradient);
	   theta[2] = actual_theta_2 - (learningRate * theta_2_gradient);
	   return theta;
}
std::vector<float>  gradient_Descent(float initial_theta_0, float initial_theta_1,float initial_theta_2, float learning_rate,int num_iterations, vector<float> &  X_1, vector<float> &  X_2, vector<float> &  Y_i){
           int i = 0;
	   int N = 4 ;
	   vector <float> theta {1,1,1};
	   theta[0] = initial_theta_0;
	   theta[1] = initial_theta_1;
	   theta[2] = initial_theta_2;
	   float Erreur_1 = 0;
	   string const nomFichier("erreur1.csv");
	   ofstream monFlux(nomFichier.c_str());
	   for (i = 0; i<num_iterations; i++){
                         theta = iteration(theta[0], theta[1],theta[2], learning_rate, X_1, X_2, Y_i, N);
			 Erreur_1 = calcul_Erreur(theta[0],theta[1],theta[2], X_1, X_2, Y_i,N);
			 monFlux << i+1 << " , " << Erreur_1 << " ; " << endl;
        
	   }
	   return theta ;
}
std::vector<float> application_Gradient_Descent(vector<float> & X_1, vector<float> & X_2, vector<float> & Y_i, int N){
           vector <float> theta {1,1,1};
	   float learning_rate = 0.001;
	   int num_iterations = 30000;
	   float initial_theta_0 = 1;
	   float initial_theta_1 = 1;
	   float initial_theta_2 = 1;
	   theta = gradient_Descent(initial_theta_0, initial_theta_1,initial_theta_2, learning_rate, num_iterations, X_1, X_2, Y_i);
	   return theta;
    
 }
std::vector<float> EstimateurMoindreCarres(vector<float> & X_1, vector<float> &  X_2, vector<float> &  Y, int N ){
           vector <float> theta {1,1,1};
	   MatrixXf X(4,3);
	   X << 1,0,0,
	     1,0,1,
	     1,1,0,
	     1,1,1;
	   MatrixXf Y_i(4,1);
	   float Erreur_1;
	   MatrixXf X_T;
	   MatrixXf inverse;
	   Y_i << 1,1,1,1;
	   X_T = X.transpose();
	   MatrixXf produit;
	   produit = X_T * X;
	   inverse = produit.inverse();
	   MatrixXf resultat;
	   resultat = inverse * X_T * Y_i;
	   theta[0] = resultat(0); theta[1] = resultat(1); theta[2] = resultat(2);
	   Erreur_1 = calcul_Erreur(theta[0], theta[1], theta[2], X_1, X_2, Y,N);
	   std::cout << "erreur optimale = "<< Erreur_1 <<std::endl;
	   return theta;
}   
int main(){
          int N=4;
	  vector <float> theta_sgd(3);
	  vector <float> theta_gd(3);
	  vector <float> theta_EMC(3);
	  vector<float> X_1 {0,0,1,1};
	  vector<float> X_2 {0,1,0,1};
	  vector<float> Y {-1,1,1,1};
	  printf("EstimateurMoindreCarres\n");
	  /* theta_EMC = EstimateurMoindreCarres(X_1, X_2, Y, N );
	  cout <<"theta_EMC[0] = " << theta_EMC[0] << endl;
	  cout <<"theta_EMC[1] = " << theta_EMC[1] << endl;
	  cout <<"theta_EMC[2] = " << theta_EMC[2] << endl;
	  printf("gradient descent\n");
	  theta_gd = application_Gradient_Descent(X_1, X_2, Y, N);
	  cout <<"theta_gd[0] = " << theta_gd[0] << endl;
	  cout <<"theta_gd[1] = " << theta_gd[1] << endl;
	  cout <<"theta_gd[2] = " << theta_gd[2] << endl;*/
	  printf("stochastic gradient descent\n");
	  theta_sgd = stochastic_gradient_descent(X_1,X_2, Y, X_1.size());
	  cout <<"theta_sgd[0] = " << theta_sgd[0] << endl;
	  cout <<"theta_sgd[1] = " << theta_sgd[1] << endl;
	  cout <<"theta_sgd[2] = " << theta_sgd[2] << endl;
	  return 1;
}    
    
