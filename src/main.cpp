#include "examples.h"
#include "homework_1.h"
#include "homework_2.h"
#include "homework_3.h"
#include "homework_4.h"
#include "homework_5.h"

#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    string homework_selection;
    cout << "Type homework number to execute: ";
    getline(cin, homework_selection);

    switch (stoi(homework_selection)) {
    case 1:
        main_homework_1();
        break;
    case 2:
        main_homework_2();
        break;
    case 3:
        main_homework_3();
        break;
    case 4:
        main_homework_4();
        break;
    case 5:
        main_homework_5();
        break;
    default:
        cout << "invalid homework selection" << endl;
        break;
    }

    return 0;
}