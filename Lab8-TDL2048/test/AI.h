#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include "2048.h"

class experience {
public:
    state sp;
    state spp;
};


class AI {
public:
    static void load_tuple_weights() {
        std::string filename = "std_tuple.weight";                   // put the name of weight file here
        std::ifstream in;
        in.open(filename.c_str(), std::ios::in | std::ios::binary);
        if (in.is_open()) {
            for (size_t i = 0; i < feature::list().size(); i++) {
                in >> *(feature::list()[i]);
                std::cout << feature::list()[i]->name() << " is loaded from " << filename << std::endl;
            }
            in.close();
        }
    }

    static void set_tuples() {
        // line
        feature::list().push_back(new pattern<4>(0, 4, 8, 12));
        feature::list().push_back(new pattern<4>(1, 5, 9, 13));
        // ax
        feature::list().push_back(new pattern<6>(0, 1, 4, 5, 8, 12));
        feature::list().push_back(new pattern<6>(1, 2, 5, 6, 9, 13));
    }

    static int get_best_move(state s) {         // return best move dir
        float best_value = 0.0;
        int best_dir = 0;
        for (int dir = 0; dir < 4; ++dir) {
            state st = s;
            int reward = st.move(dir);
            if( reward == -1 ) continue;
            float value = st.evaluate_score() + reward;
            if (value > best_value) {
                best_dir = dir;
                best_value = value;
            }
        }
        return best_dir;
    }

};