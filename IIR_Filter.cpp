#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <complex>
#include <xmmintrin.h>

using namespace std;

const double PI = 3.14159265358979323846;

// np.zeros for float
vector<float> zeros(int n) {
    return vector<float>(n, 0.0f);
}

// Structure for Second-Order Sections (SOS)
struct SOSSection {
    vector<float> b;
    vector<float> a;
};

// 2. Analog Prototype & Bilinear Transform computed in DOUBLE precision
vector<complex<double>> butterworth_analog_poles(int N, double wc) {
    vector<complex<double>> poles;
    for (int k = 0; k < N; k++) {
        double theta = (PI / 2.0) + (PI * (2 * k + 1)) / (2.0 * N);
        complex<double> p = wc * exp(complex<double>(0.0, theta));
        poles.push_back(p);
    }
    return poles;
}

///Billinear Transform mapping analog s-domain to digital z-domain
vector<complex<double>> bilinear_transform_poles(const vector<complex<double>>& poles, double T) {
    vector<complex<double>> digital_poles;
    for (auto p : poles) {
        // //Z-Plane Transform for individual poles
        complex<double> dp = (1.0 + p * T / 2.0) / (1.0 - p * T / 2.0);
        digital_poles.push_back(dp);
    }
    return digital_poles;
}

// 4. Pole ordering: Outside-in pairing for lowest-Q first stability
// //Z-Plane Transform transfer function construction & gain normalization
vector<SOSSection> build_sos_cascade(int N, const vector<complex<double>>& digital_poles) {
    vector<SOSSection> sos_sections;
    int i = 0, j = N - 1;
    while (i < j) {
        complex<double> p1 = digital_poles[i];
        complex<double> p2 = digital_poles[j];
        
        complex<double> sum_p = p1 + p2;
        complex<double> prod_p = p1 * p2;
        
        vector<double> a_sec = {1.0, -sum_p.real(), prod_p.real()};
        vector<double> b_sec = {1.0, 2.0, 1.0};
        
        double sum_a = a_sec[0] + a_sec[1] + a_sec[2];
        double sum_b = b_sec[0] + b_sec[1] + b_sec[2];
        double scale = sum_a / sum_b;
        b_sec[0] *= scale; b_sec[1] *= scale; b_sec[2] *= scale;
        
        // Cast double coefficients to float for runtime
        sos_sections.push_back({
            {static_cast<float>(b_sec[0]), static_cast<float>(b_sec[1]), static_cast<float>(b_sec[2])},
            {static_cast<float>(a_sec[0]), static_cast<float>(a_sec[1]), static_cast<float>(a_sec[2])}
        });
        i++;
        j--;
    }
    if (i == j) {
        complex<double> p = digital_poles[i];
        vector<double> a_sec = {1.0, -p.real()};
        vector<double> b_sec = {1.0, 1.0};
        
        double sum_a = a_sec[0] + a_sec[1];
        double sum_b = b_sec[0] + b_sec[1];
        double scale = sum_a / sum_b;
        b_sec[0] *= scale; b_sec[1] *= scale;
        
        sos_sections.push_back({
            {static_cast<float>(b_sec[0]), static_cast<float>(b_sec[1])},
            {static_cast<float>(a_sec[0]), static_cast<float>(a_sec[1])}
        });
    }
    return sos_sections;
}
// 3. Stateful Block-Based Processing in strict float precision
class StatefulFloat32SOSFilter {
private:
    vector<SOSSection> sos_sections;
    vector<vector<float>> x_histories;
    vector<vector<float>> y_histories;

public:
    StatefulFloat32SOSFilter(const vector<SOSSection>& sections) : sos_sections(sections) {
        for (const auto& sec : sections) {
            int b_len = sec.b.size() - 1;
            int a_len = sec.a.size() - 1;
            x_histories.push_back(zeros(b_len));
            y_histories.push_back(zeros(a_len));
        }
    }

    vector<float> process_block(const vector<float>& block) {
        vector<float> current_block = block;

        for (size_t idx = 0; idx < sos_sections.size(); idx++) {
            const auto& b = sos_sections[idx].b;
            const auto& a = sos_sections[idx].a;
            auto& x_hist = x_histories[idx];
            auto& y_hist = y_histories[idx];

            int b_len = b.size();
            int a_len = a.size();
            int hist_len = x_hist.size();

            vector<float> extended_x(hist_len + current_block.size());
            for (int k = 0; k < hist_len; k++) extended_x[k] = x_hist[k];
            vector<float> extended_y(extended_x.size(), 0.0f);
            for (int k = 0; k < hist_len; k++) extended_y[k] = y_hist[k];

            for (size_t k = 0; k < current_block.size(); k++) extended_x[hist_len + k] = current_block[k];

            // Sample-by-sample recurrence relation / //Z-Plane Transform difference equation evaluation
            for (size_t n = hist_len; n < extended_x.size(); n++) {
                float acc = 0.0f;
                for (int k = 0; k < b_len; k++) {
                    acc += b[k] * extended_x[n - k];
                }
                for (int k = 1; k < a_len; k++) {
                    acc -= a[k] * extended_y[n - k];
                }
                extended_y[n] = acc;
            }

            vector<float> filtered_block(current_block.size());
            for (size_t k = 0; k < current_block.size(); k++) {
                filtered_block[k] = extended_y[hist_len + k];
            }

            // 1. Safe history update using extended arrays (handles short blocks correctly)
            for (int k = 0; k < hist_len; k++) {
                x_hist[k] = extended_x[extended_x.size() - hist_len + k];
                y_hist[k] = extended_y[extended_y.size() - hist_len + k];
            }

            current_block = filtered_block;
        }

        return current_block;
    }
};

int main() {
    try {
        // 4. Enable SSE Flush-To-Zero to prevent denormal CPU slowdowns
        _mm_setcsr(_mm_getcsr() | 0x8040);

        // 1. Setup Synthetic Signal Parameters (replaces WAV reading)
        int sampleRate = 10000;
        double duration = 1.0;
        int num_samples = static_cast<int>(sampleRate * duration);

        vector<float> input_signal(num_samples);
        for (int i = 0; i < num_samples; i++) {
            double t = static_cast<double>(i) / sampleRate;
            // Composite signal: 10 Hz sine wave + 1500 Hz high-frequency component
            input_signal[i] = static_cast<float>(sin(2.0 * PI * 10.0 * t) + 0.5 * sin(2.0 * PI * 1500.0 * t));
        }

        cout << "Generated synthetic input signal: 1 channel at " << sampleRate << " Hz (" << num_samples << " samples)" << endl;

        // Config parameters
        float cutoff = 200.0f; // Lowpass filter cutoff
        // 4. Cutoff validation against Nyquist
        if (cutoff >= sampleRate / 2.0f) {
            throw invalid_argument("Cutoff frequency must be less than sampleRate / 2 (Nyquist).");
        }

        int N = 4;
        int block_size = 1024;
        double T = 1.0 / sampleRate;

        // 2. Frequency warping and design in double precision
        double Omega_c = 2.0 * PI * cutoff;
        double Omega_c_warped = (2.0 / T) * tan((Omega_c * T) / 2.0);

        // //DFFT (Discrete Fast Fourier Transform) frequency domain parameter preparation
        vector<complex<double>> analog_poles = butterworth_analog_poles(N, Omega_c_warped);
        vector<complex<double>> digital_poles = bilinear_transform_poles(analog_poles, T);
        vector<SOSSection> sos_sections = build_sos_cascade(N, digital_poles);

        // Process signal in blocks using stateful filter engine
        StatefulFloat32SOSFilter filter_engine(sos_sections);
        vector<float> filtered_signal;
        filtered_signal.reserve(num_samples);

        for (int start_idx = 0; start_idx < num_samples; start_idx += block_size) {
            int end_idx = min(start_idx + block_size, num_samples);
            vector<float> block(input_signal.begin() + start_idx, input_signal.begin() + end_idx);

            vector<float> filtered_block = filter_engine.process_block(block);
            filtered_signal.insert(filtered_signal.end(), filtered_block.begin(), filtered_block.end());
        }

        cout << "DIAGNOSTIC: Filtered " << filtered_signal.size() << " samples successfully with 32-bit float precision." << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}