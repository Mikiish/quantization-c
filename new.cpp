#include <iostream>
#include <vector>
using namespace std;

void afficher(const vector<int> &v) {
    for (int nombre : v)
        cout << nombre << " ";
    cout << "\n";
}

int main() {
    cout << "La suite : ";
    char ch='a'; char * ptr=&ch;
    cout << *ptr << "\n";
    cout << ptr << "\n";
    char chaine[16] = "Hello";
    vector<int> prime(5); prime = {2, 3, 5, 7, 11};
    cout << chaine << "\n";
    afficher(prime);
    prime.push_back(13);
    afficher(prime);
    while (prime.size() > 2)
        prime.pop_back();
    afficher(prime);
    return 0;
}