#include "minizero/go/go.h"
#include "minizero/utils/sgf_loader.h"
#include "minizero/utils/tqdm.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include <numeric>
#include <unordered_set>
#include <deque>
#include <iomanip>

using namespace minizero::utils;
using namespace minizero::env::go;
using minizero::env::Player;

struct MoveInfo {
    bool legal;
    std::vector<int> captured_stones;
};

struct SeqState {
    std::vector<GoAction> seq;
    GoHashKey hash;
};

struct InfoSetSizesRow {
    int move_index;
    size_t black_size;
    size_t white_size;
};


// Create GoEnv from the sequence
bool rebuildEnvFromSeq(int board_size, const std::vector<GoAction>& seq, GoEnv& env) {
    env = GoEnv(board_size);
    for (const auto& a : seq) {
        if (!env.act(a)) return false;
    }
    return true;
}

bool sameStones(const GoEnv& a, const GoEnv& b) {
    return a.getHashKey() == b.getHashKey();
}

void pruneSeqSet(std::vector<SeqState>& S, size_t max_size) {
    if (S.size() <= max_size) return;
    static std::mt19937 rng{std::random_device{}()};
    std::shuffle(S.begin(), S.end(), rng);
    S.resize(max_size);
}

MoveInfo analyzeMove(
    const GoEnv& before, 
    const GoEnv& after, 
    const GoAction& action) 
{
    MoveInfo info;
    info.legal = true;
    
    Player opponent = (action.getPlayer() == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;
    const auto& old_stones = before.getStoneBitboard();
    const auto& new_stones = after.getStoneBitboard();
    
    GoBitboard removed = old_stones.get(opponent) & ~new_stones.get(opponent);
    while (!removed.none()) {
        int pos = removed._Find_first();
        removed.reset(pos);
        info.captured_stones.push_back(pos);
    }
    
    return info;
}


bool matchesMoveInfo(
    const MoveInfo& actual, 
    const MoveInfo& expected) 
{
    if (actual.legal != expected.legal) {
        return false;
    }

    if (actual.captured_stones.size() != expected.captured_stones.size()) return false;
    
    std::vector<int> actual_sorted_caps = actual.captured_stones;
    std::vector<int> expected_sorted_caps = expected.captured_stones;
    std::sort(actual_sorted_caps.begin(), actual_sorted_caps.end());
    std::sort(expected_sorted_caps.begin(), expected_sorted_caps.end());
    if (actual_sorted_caps != expected_sorted_caps) return false;

    return true;
}

std::vector<SeqState> findMatchingNextSeqs(
    const std::vector<GoAction>& current_seq,
    const MoveInfo& referee_info,
    Player moving_player,
    int board_size)
{
    std::vector<SeqState> out;
    GoEnv current_board(board_size);
    if (!rebuildEnvFromSeq(board_size, current_seq, current_board)) return out;

    // try every possible positions
    for (int pos = 0; pos < board_size * board_size; ++pos) {
        GoEnv test_board = current_board;
        GoAction guess_action(pos, moving_player);
        if (test_board.act(guess_action)) {
            MoveInfo guess_info = analyzeMove(current_board, test_board, guess_action);
            if (matchesMoveInfo(guess_info, referee_info)) {
                SeqState s;
                s.seq = current_seq;
                s.seq.push_back(guess_action);
                s.hash = test_board.getHashKey();
                out.push_back(std::move(s));
            }
        }
    }

    // Consider PASS
    {
        GoEnv test_board = current_board;
        GoAction pass_action(board_size * board_size, moving_player);
        if (test_board.act(pass_action)) {
            MoveInfo pass_info = analyzeMove(current_board, test_board, pass_action);
            if (matchesMoveInfo(pass_info, referee_info)) {
                SeqState s;
                s.seq = current_seq;
                s.seq.push_back(pass_action);
                s.hash = test_board.getHashKey();
                out.push_back(std::move(s));
            }
        }
    }
    return out;
}

std::vector<SeqState> generateInfoSet(
    const std::vector<SeqState>& previous_info_set,
    const GoEnv& referee_board_before,
    const GoAction& referee_action)
{
    GoEnv referee_board_after = referee_board_before;
    if (!referee_board_after.act(referee_action)) {
        std::cerr << "CRITICAL ERROR in generateInfoSet: Real action illegal.\n";
        return {};
    }
    MoveInfo true_referee_info = analyzeMove(referee_board_before, referee_board_after, referee_action);
    Player moving_player = referee_action.getPlayer();
    int board_size = referee_board_before.getBoardSize();

    std::vector<SeqState> new_info_set;
    new_info_set.reserve(previous_info_set.size() << 2);
    std::unordered_set<GoHashKey> seen;

    for (const auto& cand : previous_info_set) {
        auto nexts = findMatchingNextSeqs(cand.seq, true_referee_info, moving_player, board_size);
        for (auto& s : nexts) {
            if (seen.insert(s.hash).second) new_info_set.emplace_back(std::move(s));
        }
    }
    return new_info_set;
}

std::vector<SeqState> updateInfoSetForActor(
    const std::vector<SeqState>& previous_info_set,
    const MoveInfo& expected_info,
    const GoAction& actor_action,
    int board_size)
{
    std::vector<SeqState> out;
    out.reserve(previous_info_set.size());
    std::unordered_set<GoHashKey> seen;

    for (const auto& cand : previous_info_set) {
        GoEnv before(board_size);
        if (!rebuildEnvFromSeq(board_size, cand.seq, before)) continue;

        GoEnv after = before;
        if (!after.act(actor_action)) continue;

        MoveInfo mi = analyzeMove(before, after, actor_action);

        if (!matchesMoveInfo(mi, expected_info)) continue;

        SeqState s;
        s.seq  = cand.seq;
        s.seq.push_back(actor_action);
        s.hash = after.getHashKey();

        if (seen.insert(s.hash).second) out.emplace_back(std::move(s));
    }
    return out;
}



std::vector<SeqState> findInfoSetAtMove(
    Player target_player,
    int move_number,
    const std::vector<GoEnv>& referee_history,
    const std::vector<GoAction>& referee_actions,
    size_t max_info_set_size, 
    std::vector<InfoSetSizesRow>* sizes_log)
{
    if (move_number < 0 || move_number > (int)referee_actions.size()) {
        std::cerr << "Error: Invalid move_number " << move_number << "\n";
        return {};
    }
    const int board_size = referee_history[0].getBoardSize();

    std::vector<SeqState> black_set, white_set;
    {
        GoEnv empty(board_size);
        SeqState root{ {}, empty.getHashKey() };
        black_set = { root };
        white_set = { root };
    }

    std::vector<int> moves_to_process(move_number);
    std::iota(moves_to_process.begin(), moves_to_process.end(), 1);

    auto pbar = minizero::utils::tqdm::tqdm(
        moves_to_process,
        "Calculating: {step}/{size}"
    );

    for (int i : pbar) {
        int k = i - 1;
        const GoAction& real_action = referee_actions[k];
        const int board_size = referee_history[0].getBoardSize();

        GoEnv ref_before = referee_history[k];
        GoEnv ref_after  = ref_before;
        bool ok = ref_after.act(real_action);
        if (!ok) { std::cerr << "CRITICAL: real action illegal\n"; break; }
        MoveInfo true_info = analyzeMove(ref_before, ref_after, real_action);

        Player mover = real_action.getPlayer();
        if (mover == Player::kPlayer1) {
            black_set = updateInfoSetForActor(black_set, true_info, real_action, board_size);
            white_set = generateInfoSet(white_set, referee_history[k], real_action);
        } else {
            white_set = updateInfoSetForActor(white_set, true_info, real_action, board_size);
            black_set = generateInfoSet(black_set, referee_history[k], real_action);
        }

        pruneSeqSet(black_set, max_info_set_size);
        pruneSeqSet(white_set, max_info_set_size);

        if (sizes_log) {
            sizes_log->push_back(InfoSetSizesRow{
                .move_index = i,
                .black_size = black_set.size(),
                .white_size = white_set.size()
        });
    }
    }


    return (target_player == Player::kPlayer1) ? black_set : white_set;
}

void analyzeAndDisplayInfoSet(
    Player player,
    int move_number,
    const std::vector<GoEnv>& referee_history,
    const std::vector<GoAction>& referee_actions,
    size_t max_info_set_size)
{
    if (move_number <= 0 || move_number > (int)referee_actions.size()) {
        std::cerr << "Error: Invalid move number " << move_number << "\n";
        return;
    }
    const int board_size = referee_history[0].getBoardSize();
    std::string player_name = (player == Player::kPlayer1) ? "Black" : "White";

    std::cout << "\n=================================================================\n"
              << "Calculating Information Set for " << player_name
              << " after move #" << move_number << "...\n"
              << "=================================================================\n\n";

    
    std::vector<InfoSetSizesRow> sizes_log;
    std::vector<SeqState> seq_set = findInfoSetAtMove(
        player, move_number, referee_history, referee_actions, max_info_set_size, &sizes_log);

    // print the real board
    const GoEnv& truth = referee_history[move_number];
    std::cout << "\n[Real referee board after move #" << move_number << "]\n" << truth.toString() << "\n";

    // pick 5 boards from the information set randomly
    {
        static std::mt19937 rng{std::random_device{}()};
        std::vector<size_t> idx(seq_set.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        size_t show = std::min<size_t>(5, idx.size());
        for (size_t i = 0; i < show; ++i) {
            GoEnv env(board_size);
            rebuildEnvFromSeq(board_size, seq_set[idx[i]].seq, env);
            std::cout << "Possibility #" << (i+1) << ":\n" << env.toString() << "\n";
        }
    }

    // Check if the real board in the information set
    bool present = false;
    for (const auto& s : seq_set) {
        GoEnv env(board_size);
        if (!rebuildEnvFromSeq(board_size, s.seq, env)) continue;
        if (sameStones(env, truth)) { present = true; break; }
    }

    std::cout << "\nPer-move Information Set Sizes\n";

    const int W1 = 6, W2 = 12, W3 = 12;
    std::cout << std::left  << std::setw(W1) << "Move"
            << std::right << std::setw(W2) << "Black"
            << std::setw(W3) << "White" << '\n';

    for (const auto& row : sizes_log) {
        std::cout << std::left  << std::setw(W1) << row.move_index
                << std::right << std::setw(W2) << row.black_size
                << std::setw(W3) << row.white_size << '\n';
    }

    std::cout << "\n---------------------------\n"
              << "Check if the referee's board is present in InfoSet\n"
              << "---------------------------\n";
    if (present) {
        std::cout << "SUCCESS: Real board (by stones) is in the information set.\n";
    } else {
        std::cout << "WARNING: Real board (by stones) NOT found in the information set.\n";
    }
}



int main() {
    SGFLoader loader;
    if (!loader.loadFromFile("../test.sgf")) {
        std::cerr << "Error: Cannot load .sgf file" << std::endl;
        return 1;
    }
    
    std::cout << "Sucessfully load .sgf file; totally " << loader.getActions().size() << " moves" << std::endl;

    int board_size = 9;
    minizero::config::env_board_size = board_size;
    initialize();
    const size_t MAX_INFO_SET_SIZE = 100000;

    std::vector<GoEnv> referee_history;
    std::vector<GoAction> referee_actions;
    
    GoEnv current_env(board_size);
    referee_history.push_back(current_env);

    for (const auto& action_pair : loader.getActions()) {
        const auto& sgf_action = action_pair.first;
        if (sgf_action.size() < 2) continue;
        
        Player player = (sgf_action[0] == "B") ? Player::kPlayer1 : Player::kPlayer2;
        int action_id;
        if (sgf_action[1].empty() || sgf_action[1] == "tt" || sgf_action[1] == "pass") {
            action_id = board_size * board_size;
        } else {
            action_id = SGFLoader::boardCoordinateStringToActionID(sgf_action[1], board_size);
        }
        
        GoAction action(action_id, player);
        if (!current_env.act(action)) {
            std::cerr << "Invalid move: " << referee_actions.size() + 1 << std::endl;
            return 1;
        }
        
        referee_actions.push_back(action);
        referee_history.push_back(current_env);
    }

    int target_move = 14;
    analyzeAndDisplayInfoSet(Player::kPlayer2, target_move, referee_history, referee_actions, MAX_INFO_SET_SIZE);

    return 0;
}