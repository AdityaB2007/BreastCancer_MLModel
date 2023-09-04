#include <iostream>

void drink(int cups){
  for (int i = 0 ; i < cups; i++){
    std::cout << "You need to drink more water!" << std::endl;
  }
}

int main(){
  drink(8);
}