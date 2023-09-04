#include <iostream>

void drink(int cups){
  if (cups > 0){
    std::cout << "You need to drink more water!" << std::endl;
    drink(cups - 1);
  }
}

int main(){
  drink(8);
}