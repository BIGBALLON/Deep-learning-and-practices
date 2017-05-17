/**
* Temporal Difference Learning Demo for Game 2048
* use 'g++ -O3 -o 2048 2048.cpp' to compile the source
*
* Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
* http://www.aigames.nctu.edu.tw/
* January 2017
*
* References:
* [1] Szubert, Marcin, and Wojciech Jaśkowski. "Temporal difference learning of n-tuple networks for the game 2048."
* Computational Intelligence and Games (CIG), 2014 IEEE Conference on. IEEE, 2014.
* [2] Wu, I-Chen, et al. "Multi-stage temporal difference learning for 2048."
* Technologies and Applications of Artificial Intelligence. Springer International Publishing, 2014. 366-378.
*/
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include <ctime>
#include "2048.h"
#include "AI.h"

const int TEST_COUNT = 10000;

int main(int argc, const char* argv[]) {
    std::srand(std::time(NULL));

    AI::set_tuples();                       // set agent tuples
    AI::load_tuple_weights();               // load tuple weights from file

    int scores[TEST_COUNT];                 // for statistics
    int maxtile[TEST_COUNT];

    for (int n = 0; n < TEST_COUNT; n++) {
        int a;
        // play an episode
        int score = 0;
        board b;
        b.init();

        // std::cout << "Game " << n << std::endl;

        while (true) {
            // try to find a best move
            state current_state(b);
            int best_move = AI::get_best_move(current_state);
            state best_next_state = current_state;
            best_next_state.move(best_move);

            if (best_next_state.get_reward() != -1) {
                b = best_next_state.get_board();
                score += best_next_state.get_reward();
                b.add_random_tile();
            }
            // game over
            else {
                break;
            }
        }

        scores[n] = score;
        maxtile[n] = 0;
        for (int i = 0; i < 16; i++)
            maxtile[n] = std::max(maxtile[n], b.at(i));
        
    }

    int success_count = 0;
    for (int i = 0; i < TEST_COUNT; i++) {
        if (maxtile[i] >= 11)
            success_count++;
    }
    std::cout << "Success Rate:\t" << float(success_count) / float(TEST_COUNT) << std::endl;

    return 0;
}
