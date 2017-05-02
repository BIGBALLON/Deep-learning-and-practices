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
#include <cstring>
/**
* The simplest bitboard implementation for 2048 board
*
* Index Mapping:
* -------------
*  0  1  2  3
* -------------
*  4  5  6  7
* -------------
*  8  9  10 11
* -------------
*  12 13 14 15
* -------------
*
*/
class board {
public:
	typedef unsigned long long value_t;

	inline board(const value_t& raw = 0) : raw(raw) {}
	inline board(const board& b) : raw(b.raw) {}
	inline operator value_t&() { return raw; }

	inline int  fetch(const int& i) const { return ((raw >> (i << 4)) & 0xffff); }			// fetch row
	inline void place(const int& i, const int& r) { raw = (raw & ~(0xffffULL << (i << 4))) | (value_t(r & 0xffff) << (i << 4)); }
	inline int  at(const int& i) const { return (raw >> (i << 2)) & 0x0f; }
	inline void set(const int& i, const int& t) { raw = (raw & ~(0x0fULL << (i << 2))) | (value_t(t & 0x0f) << (i << 2)); }

private:
	struct lookup {
		int raw;	// base row (16-bit raw)
		int left;	// left operation
		int right;	// right operation
		int score;	// merge reward

		void init(const int& r) {
			raw = r;

			int V[4] = { (r >> 0) & 0x0f, (r >> 4) & 0x0f, (r >> 8) & 0x0f, (r >> 12) & 0x0f };
			int L[4] = { V[0], V[1], V[2], V[3] };
			int R[4] = { V[3], V[2], V[1], V[0] }; // mirrored

			score = mvleft(L);
			left = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12));

			score = mvleft(R); std::reverse(R, R + 4);
			right = ((R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12));
		}

		inline void move_left(value_t& raw, int& sc, const int& i) const {
			raw |= value_t(left) << (i << 4);
			sc += score;
		}

		inline void move_right(value_t& raw, int& sc, const int& i) const {
			raw |= value_t(right) << (i << 4);
			sc += score;
		}

		static int mvleft(int row[]) {
			int top = 0;
			int tmp = 0;
			int score = 0;

			for (int i = 0; i < 4; i++) {
				int tile = row[i];
				if (tile == 0) continue;
				row[i] = 0;
				if (tmp != 0) {
					if (tile == tmp) {
						tile = tile + 1;
						row[top++] = tile;
						score += (1 << tile);
						tmp = 0;
					}
					else {
						row[top++] = tmp;
						tmp = tile;
					}
				}
				else {
					tmp = tile;
				}
			}
			if (tmp != 0) row[top] = tmp;
			return score;
		}

		struct init_t {
			init_t(lookup* c) {
				for (size_t i = 0; i < 65536; i++)
					c[i].init(i);
			}
		};

		static const lookup& find(const int& row) {
			static lookup cache[65536];
			static init_t init(cache);
			return cache[row];
		}
	};

public:
	inline int move_left() {
		value_t move = 0;
		value_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_left(move, score, 0);
		lookup::find(fetch(1)).move_left(move, score, 1);
		lookup::find(fetch(2)).move_left(move, score, 2);
		lookup::find(fetch(3)).move_left(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	inline int move_right() {
		value_t move = 0;
		value_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_right(move, score, 0);
		lookup::find(fetch(1)).move_right(move, score, 1);
		lookup::find(fetch(2)).move_right(move, score, 2);
		lookup::find(fetch(3)).move_right(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	inline int move_up() {
		rotate_right();
		int score = move_right();
		rotate_left();
		return score;
	}
	inline int move_down() {
		rotate_right();
		int score = move_left();
		rotate_left();
		return score;
	}
	inline int move(const int& opcode) { // 0:up 1:right 2:down 3:left
		switch (opcode) {
		case 0: return move_up();
		case 1: return move_right();
		case 2: return move_down();
		case 3: return move_left();
		default: return move((opcode % 4 + 4) % 4);
		}
	}

	inline void transpose() {
		raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
		raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
	}
	inline void mirror() {
		raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
			| ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
	}
	inline void flip() {
		raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
			| ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
	}

	inline void rotate_right() { transpose(); mirror(); }	// clockwise
	inline void rotate_left() { transpose(); flip(); }		// counterclockwise
	inline void reverse() { mirror(); flip(); }

	inline void rotate(const int& r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	inline void init() { raw = 0; add_random_tile(); add_random_tile(); }
	inline void add_random_tile() { // add a new random 2-tile or 4-tile
		int space[16], num = 0;
		for (int i = 0; i < 16; i++)
			if (at(i) == 0) {
				space[num++] = i;
			}
		if (num)
			set(space[rand() % num], rand() % 10 ? 1 : 2);
	}

	friend std::ostream& operator <<(std::ostream& out, const board& b) {
		char buff[32];
		out << "+------------------------+" << std::endl;
		for (int i = 0; i < 16; i += 4) {
			snprintf(buff, sizeof(buff), "|%6u%6u%6u%6u|",
				(1 << b.at(i + 0)) & -2u,
				(1 << b.at(i + 1)) & -2u,
				(1 << b.at(i + 2)) & -2u,
				(1 << b.at(i + 3)) & -2u);
			out << buff << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

private:
	value_t raw;
};

/**
* feature and weight table for temporal difference learning
*/
class feature {
public:
	feature(const size_t& len) : length(len), weight(alloc(len)) {}
	virtual ~feature() { delete[] weight; }
	inline float& operator[] (const size_t& i) { return weight[i]; }
	size_t size() const { return length; }
	static std::vector<feature*>& list() {
		static std::vector<feature*> feats;
		return feats;
	}
	friend std::ostream& operator <<(std::ostream& out, const feature& w) {
		std::string name = w.name();
		int len = name.length();
		out.write(reinterpret_cast<char*>(&len), sizeof(int));
		out.write(name.c_str(), len);
		float* weight = w.weight;
		size_t size = w.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size_t));
		out.write(reinterpret_cast<char*>(weight), sizeof(float) * size);
		return out;
	}
	friend std::istream& operator >> (std::istream& in, feature& w) {
		std::string name;
		int len = 0;
		in.read(reinterpret_cast<char*>(&len), sizeof(int));
		name.resize(len);
		in.read(&name[0], len);
		if (name != w.name()) {
			std::cerr << "unexpected feature: " << name << " (" << name << " is expected)" << std::endl;
			std::exit(1);
		}
		float* weight = w.weight;
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		if (size != w.size()) std::exit(1);
		in.read(reinterpret_cast<char*>(weight), sizeof(float) * size);
		if (!in) {
			std::cerr << "unexpected end of binary" << std::endl;
			std::exit(1);
		}
		return in;
	}
public: // should be implemented
	virtual float estimate(const board& b) = 0;
	virtual float update(const board& b, const float& upd) = 0;
	virtual std::string name() const = 0;
protected:
	static float* alloc(size_t num) {
		static size_t total = 0;
		static size_t limit = 1 << 30; // 1G memory
		try {
			total += num;
			if (total > limit) throw std::bad_alloc();
			return new float[num]();
		}
		catch (std::bad_alloc&) {
			std::cerr << "memory limit exceeded" << std::endl;
			std::exit(-1);
		}
		return NULL;
	}
	size_t length;
	float* weight;
};

/**
* the pattern feature
*/
template<int N>
class pattern : public feature {
public:
	pattern(int t0, ...) : feature(1 << (N * 4)) {
		va_list ap;
		va_start(ap, t0);
		patt[0] = t0;
		for (int n = 1; n < N; n++) {
			patt[n] = va_arg(ap, int);
		}
		va_end(ap);

		int isopatt[N];
		for (int i = 0; i < 8; i++) { // rotate and mirror the pattern
			board iso = 0xfedcba9876543210ull;
			if (i >= 4) iso.mirror();
			iso.rotate(i);
			for (int n = 0; n < N; n++)
				isopatt[n] = iso.at(patt[n]);
			isomorphic[i].init(isopatt);
		}

		std::cout << name() << " initialized, size = " << length;
		if (length >= (1 << 30)) {
			std::cout << " (" << (length >> 30) << "G)";
		}
		else if (length >= (1 << 20)) {
			std::cout << " (" << (length >> 20) << "M)";
		}
		else if (length >= (1 << 10)) {
			std::cout << " (" << (length >> 10) << "K)";
		}
		std::cout << std::endl;
	}
	virtual ~pattern() {}

	virtual float estimate(const board& b) {
		//		std::cout << name() << " estimate: " << std::endl << b;
		float value = 0;
		for (int i = 0; i < 8; i++)
		//	value += (operator [](isomorphic[i][b]));
			value += (*this)[isomorphic[i][b]];
		return value;
	}
	virtual float update(const board& b, const float& v) {
		//		std::cout << name() << " update: " << v << std::endl;
		float value = 0;
		for (int i = 0; i < 8; i++)
			value += (operator [](isomorphic[i][b]) += v);
		return value;
	}
	virtual std::string name() const {
		std::stringstream ss;
		ss << N << "-tuple pattern " << std::hex;
		for (int i = 0; i < N; i++)
			ss << patt[i];
		return ss.str();
	}
private:
	struct indexer {
		int patt[N];
		void init(int p[N]) { std::copy(p, p + N, patt); }
		inline size_t operator[](const board& b) const {
			size_t index = 0;
			for (int i = 0; i < N; i++)
				index |= b.at(patt[i]) << (4 * i);
			return index;
		}
		std::string name() const {
			std::stringstream ss;
			ss << std::hex;
			for (int i = 0; i < N; i++)
				ss << patt[i];
			return ss.str();
		}
	};

	int patt[N];
	indexer isomorphic[8];
};

/**
* afterstate wrapper
*/
class state {
public:
	state() : opcode(-1), after(0), value(0), reward(-1) {}
	state(const board& b) : opcode(-1), after(b), value(0), reward(-1) {}
	state(const int& opcode) : opcode(opcode), after(0), value(0), reward(-1) {}
	state(const state& st) : opcode(st.opcode), after(st.after), value(st.value), reward(st.reward) {}
	operator board& () { return after; }
	operator int& () { return reward; }

	bool operator >(const state& st) const { return value > st.value; }
	int move(const int opcode) {			// return reward
		reward = after.move(opcode);
		return reward;
	}

	int get_reward() { return reward; }

	int evaluate_score() {
		value = 0;
		for (size_t i = 0; i < feature::list().size(); i++)
			value += feature::list()[i]->estimate(after);
		return value;
	}

	board get_board() { return after; }

	bool is_valid() const {
		if (std::isnan(value)) {
			std::cerr << "numeric exception" << std::endl;
			std::exit(1);
		}
		return reward != -1;
	}
	const char* name() const {
		static const char* opname[4] = { "up", "right", "down", "left" };
		return opname[opcode];
	}

	friend std::ostream& operator <<(std::ostream& out, const state& st) {
		out << "moving " << st.name() << ", reward = " << st.reward;
		if (st.is_valid()) {
			std::cout << ", value = " << st.value << std::endl << st.after;
		}
		else {
			std::cout << " (invalid)" << std::endl;
		}
		return out;
	}
private:
	int opcode;
	board after;
	float value;
	int reward;
};

class experience {
public:
	state sp;
	state spp;
};

class AI {
public:
	static int get_best_move(state s) {			// return best move dir
//-------------TO DO--------------------------------
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
//--------------------------------------------------
	}


	static void update_tuple_values(std::vector<experience> eb, float learning_rate) {
		for (int i = eb.size() - 1; i >= 0; i--) {
			float error = 0.0;
//-------------TO DO--------------------------------
			if (i == eb.size() - 1) {
				// calculate TD error
				error = 0.0 - eb[i].sp.evaluate_score();
			}
			else {
				// calculate TD error
				error = eb[i].sp.get_reward() + eb[i+1].sp.evaluate_score()
				- eb[i].sp.evaluate_score();
				
			}
//--------------------------------------------------
			for (size_t j = 0; j < feature::list().size(); j++)
				feature::list()[j]->update(eb[i].sp.get_board(), error * learning_rate);
		}
	}
};

int main(int argc, const char* argv[]) {
	unsigned int seed;
	__asm__ __volatile__("rdtsc" : "=a" (seed));
	std::srand(seed);
	std::string load;
	std::string save;
	// initialize the learning weight and patterns
//-------------TO DO--------------------------------
	// set tuple features
	// example: feature::list().push_back(new pattern<4>(0, 1, 2, 3));

	if( strcmp(argv[1],"four") == 0 ){
		load = "4_tuple.weight";
		save = "4_tuple.weight";
		feature::list().push_back(new pattern<4>(0, 4, 8, 12));
		feature::list().push_back(new pattern<4>(1, 5, 9, 13));
	}else{
		load = "6_tuple.weight";
		save = "6_tuple.weight";
		feature::list().push_back(new pattern<6>(0, 1, 4, 5, 8, 12));
		feature::list().push_back(new pattern<6>(1, 2, 5, 6, 9, 13));

	}
	
//--------------------------------------------------

	std::vector<experience> exp_buffer;
	exp_buffer.reserve(20000);
	int scores[1000];						// for statistics
	int maxtile[1000];

	// load weights from binary file
	



	std::ifstream in;
	in.open(load.c_str(), std::ios::in | std::ios::binary);
	if (in.is_open()) {
		for (size_t i = 0; i < feature::list().size(); i++) {
			in >> *(feature::list()[i]);
			std::cout << feature::list()[i]->name() << " is loaded from " << load << std::endl;
		}
		in.close();
	}
	

	for (int n = 0; n < 5000000; n++) {
		int a;
		// play an episode
		int score = 0;
		board b;
		b.init();

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
				experience e;
				e.sp = best_next_state;
				e.spp = state(b);
				exp_buffer.push_back(e);
			}
			// game over
			else {
				break;
			}
		}

		AI::update_tuple_values(exp_buffer, 0.0025);
		exp_buffer.clear();

		// statistics
		int ep = n % 1000;
		scores[ep] = score;
		maxtile[ep] = 0;
		for (int i = 0; i < 16; i++)
			maxtile[ep] = std::max(maxtile[ep], b.at(i));

		// show the training process
		if ((n + 1) % 10000 == 0) {
			float sum = 0;
			int max = 0;
			int stat[16] = { 0 };
			for (int i = 0; i < 1000; i++) {
				sum += scores[i];
				max = std::max(max, scores[i]);
				stat[maxtile[i]]++;
			}
			float mean = sum / 1000;
			std::cout << (n + 1);
			std::cout << "\t" "mean = " << mean;
			std::cout << "\t" "max = " << max;
			std::cout << std::endl;

			int t = 1;
			while (stat[t] == 0) t++;
			for (int c = 0; c < 1000; t++) {
				c += stat[t];
				std::cout << "\t" << ((1 << t) & -2u) << "\t" << (stat[t] * 0.1) << "%\t(" << (c * 0.1) << "%)" << std::endl;
			}
		}

		// save weight table to binary file
		if ((n + 1) % 150000 == 0) {
			std::ofstream out;
			out.open(save.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
			if (out.is_open()) {
				for (size_t i = 0; i < feature::list().size(); i++) {
					out << *(feature::list()[i]);
					//std::cout << feature::list()[i]->name() << " is saved to " << save << std::endl;
				}
				out.flush();
				out.close();
			}
		}
		
	}
	return 0;
}
