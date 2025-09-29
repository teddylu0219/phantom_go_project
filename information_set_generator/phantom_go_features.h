// phantom_go_features.h
#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <fstream>

namespace PhantomGo {

struct TimeStepObservation {
    std::array<std::array<uint8_t, 9>, 9> self_stones;
    std::array<std::array<uint8_t, 9>, 9> illegal_attempts;
    std::array<std::array<uint8_t, 9>, 9> legal_move;
    std::array<std::array<uint8_t, 9>, 9> captured_black;
    std::array<std::array<uint8_t, 9>, 9> captured_white;
    uint8_t is_pass;
    
    TimeStepObservation() { reset(); }
    
    void reset() {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                self_stones[i][j] = 0;
                illegal_attempts[i][j] = 0;
                legal_move[i][j] = 0;
                captured_black[i][j] = 0;
                captured_white[i][j] = 0;
            }
        }
        is_pass = 0;
    }
};

struct AnchorFeatures {
    static constexpr int HISTORY_LENGTH = 12;
    static constexpr int BOARD_SIZE = 9;
    
    std::vector<TimeStepObservation> history;
    
    AnchorFeatures() {
        history.resize(HISTORY_LENGTH);
    }
    
    std::vector<float> toFlatArray() const {
        std::vector<float> result(HISTORY_LENGTH * 6 * BOARD_SIZE * BOARD_SIZE, 0.0f);
        
        for (size_t t = 0; t < history.size(); ++t) {
            const auto& obs = history[t];
            int base_idx = t * 6 * BOARD_SIZE * BOARD_SIZE;
            
            for (int i = 0; i < BOARD_SIZE; ++i) {
                for (int j = 0; j < BOARD_SIZE; ++j) {
                    int idx = i * BOARD_SIZE + j;
                    
                    result[base_idx + idx] = obs.self_stones[i][j];
                    result[base_idx + 81 + idx] = obs.illegal_attempts[i][j];
                    result[base_idx + 162 + idx] = obs.legal_move[i][j];
                    result[base_idx + 243 + idx] = obs.captured_black[i][j];
                    result[base_idx + 324 + idx] = obs.captured_white[i][j];
                    result[base_idx + 405 + idx] = obs.is_pass;
                }
            }
        }
        
        return result;
    }
};

struct BoardState {
    static constexpr int BOARD_SIZE = 9;
    
    std::array<std::array<uint8_t, BOARD_SIZE>, BOARD_SIZE> black_stones;
    std::array<std::array<uint8_t, BOARD_SIZE>, BOARD_SIZE> white_stones;
    
    BoardState() { reset(); }
    
    void reset() {
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                black_stones[i][j] = 0;
                white_stones[i][j] = 0;
            }
        }
    }
    
    std::vector<float> toFlatArray() const {
        std::vector<float> result(2 * BOARD_SIZE * BOARD_SIZE);
        
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                result[i * BOARD_SIZE + j] = black_stones[i][j];
                result[81 + i * BOARD_SIZE + j] = white_stones[i][j];
            }
        }
        
        return result;
    }
    
    bool equals(const BoardState& other) const {
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (black_stones[i][j] != other.black_stones[i][j] ||
                    white_stones[i][j] != other.white_stones[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
};

struct TrainingSample {
    AnchorFeatures anchor;
    BoardState true_board;
    std::vector<BoardState> false_boards;
    
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buffer;
        
        // 1. Anchor
        auto anchor_data = anchor.toFlatArray();
        buffer.insert(buffer.end(), 
                     reinterpret_cast<const uint8_t*>(anchor_data.data()),
                     reinterpret_cast<const uint8_t*>(anchor_data.data()) + 
                     anchor_data.size() * sizeof(float));
        
        // 2. True board
        auto true_data = true_board.toFlatArray();
        buffer.insert(buffer.end(),
                     reinterpret_cast<const uint8_t*>(true_data.data()),
                     reinterpret_cast<const uint8_t*>(true_data.data()) + 
                     true_data.size() * sizeof(float));
        
        // 3. Number of false boards
        uint32_t num_false = false_boards.size();
        buffer.insert(buffer.end(),
                     reinterpret_cast<const uint8_t*>(&num_false),
                     reinterpret_cast<const uint8_t*>(&num_false) + sizeof(uint32_t));
        
        // 4. False boards
        for (const auto& board : false_boards) {
            auto board_data = board.toFlatArray();
            buffer.insert(buffer.end(),
                         reinterpret_cast<const uint8_t*>(board_data.data()),
                         reinterpret_cast<const uint8_t*>(board_data.data()) + 
                         board_data.size() * sizeof(float));
        }
        
        return buffer;
    }
};

} // namespace PhantomGo